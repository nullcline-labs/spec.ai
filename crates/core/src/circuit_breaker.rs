use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use tokio::sync::Mutex;

/// Circuit breaker states represented as u8 for atomic storage.
const STATE_CLOSED: u8 = 0;
const STATE_OPEN: u8 = 1;
const STATE_HALF_OPEN: u8 = 2;

/// Error returned when the circuit breaker is open and rejecting requests.
#[derive(Debug, thiserror::Error)]
#[error("Circuit breaker is open; requests are being rejected")]
pub struct CircuitOpenError;

/// A generic async-safe circuit breaker.
///
/// Transitions:
/// - **Closed** (normal): requests flow through. After `failure_threshold`
///   consecutive failures the breaker moves to Open.
/// - **Open** (rejecting): all requests are immediately rejected with
///   [`CircuitOpenError`]. After `recovery_timeout` the breaker moves to
///   HalfOpen.
/// - **HalfOpen** (testing): a single request is allowed through. If it
///   succeeds the breaker resets to Closed; if it fails the breaker returns
///   to Open.
pub struct CircuitBreaker {
    state: AtomicU8,
    consecutive_failures: AtomicU64,
    failure_threshold: u64,
    recovery_timeout: std::time::Duration,
    last_failure: Mutex<Option<std::time::Instant>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// * `failure_threshold` – number of consecutive failures before opening.
    /// * `recovery_timeout` – duration to wait before transitioning from Open
    ///   to HalfOpen.
    pub fn new(failure_threshold: u64, recovery_timeout: std::time::Duration) -> Self {
        Self {
            state: AtomicU8::new(STATE_CLOSED),
            consecutive_failures: AtomicU64::new(0),
            failure_threshold,
            recovery_timeout,
            last_failure: Mutex::new(None),
        }
    }

    /// Check whether a request is allowed to proceed.
    ///
    /// Returns `Ok(())` when the circuit is Closed or transitions to HalfOpen.
    /// Returns `Err(CircuitOpenError)` when the circuit is Open and the
    /// recovery timeout has not yet elapsed.
    pub async fn check(&self) -> Result<(), CircuitOpenError> {
        let state = self.state.load(Ordering::SeqCst);

        match state {
            STATE_CLOSED => Ok(()),
            STATE_HALF_OPEN => {
                // Allow exactly one probe request through.
                Ok(())
            }
            STATE_OPEN => {
                let last = self.last_failure.lock().await;
                if let Some(instant) = *last {
                    if instant.elapsed() >= self.recovery_timeout {
                        drop(last);
                        self.state.store(STATE_HALF_OPEN, Ordering::SeqCst);
                        tracing::warn!("Circuit breaker transitioning to HalfOpen");
                        return Ok(());
                    }
                }
                Err(CircuitOpenError)
            }
            _ => unreachable!("Invalid circuit breaker state"),
        }
    }

    /// Record a successful request. Resets the breaker to Closed.
    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::SeqCst);
        self.state.store(STATE_CLOSED, Ordering::SeqCst);
    }

    /// Record a failed request. If the consecutive failure count reaches the
    /// threshold the breaker opens.
    pub async fn record_failure(&self) {
        let prev = self.consecutive_failures.fetch_add(1, Ordering::SeqCst);
        let new_count = prev + 1;

        {
            let mut last = self.last_failure.lock().await;
            *last = Some(std::time::Instant::now());
        }

        if new_count >= self.failure_threshold {
            let previous_state = self.state.swap(STATE_OPEN, Ordering::SeqCst);
            if previous_state != STATE_OPEN {
                tracing::warn!(
                    consecutive_failures = new_count,
                    "Circuit breaker opened after {} consecutive failures",
                    new_count
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_closed_allows_requests() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(10));
        assert!(cb.check().await.is_ok());
        assert!(cb.check().await.is_ok());
        assert!(cb.check().await.is_ok());
    }

    #[tokio::test]
    async fn test_opens_after_threshold_failures() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(10));

        cb.record_failure().await;
        assert!(
            cb.check().await.is_ok(),
            "Should still be closed after 1 failure"
        );

        cb.record_failure().await;
        assert!(
            cb.check().await.is_ok(),
            "Should still be closed after 2 failures"
        );

        cb.record_failure().await;
        // After 3 failures (threshold=3), the breaker should be open.
        assert!(cb.check().await.is_err(), "Should be open after 3 failures");
    }

    #[tokio::test]
    async fn test_open_rejects_requests() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(60));

        cb.record_failure().await;
        cb.record_failure().await;

        assert!(cb.check().await.is_err());
        assert!(cb.check().await.is_err());
        assert!(cb.check().await.is_err());
    }

    #[tokio::test]
    async fn test_half_open_after_recovery_timeout() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(50));

        cb.record_failure().await;
        assert!(cb.check().await.is_err(), "Should be open immediately");

        // Wait for recovery timeout to elapse.
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Should now transition to HalfOpen and allow a request.
        assert!(
            cb.check().await.is_ok(),
            "Should be half-open after recovery timeout"
        );
    }

    #[tokio::test]
    async fn test_success_resets_to_closed() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(50));

        // Trip the breaker.
        cb.record_failure().await;
        assert!(cb.check().await.is_err());

        // Wait for recovery.
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Transition to HalfOpen.
        assert!(cb.check().await.is_ok());

        // Simulate a successful probe.
        cb.record_success();

        // Should now be Closed and allow requests freely.
        assert!(cb.check().await.is_ok());
        assert!(cb.check().await.is_ok());
    }
}
