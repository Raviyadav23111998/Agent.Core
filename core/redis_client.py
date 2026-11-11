"""
Redis client — shared distributed-state backend (roadmap K).
═══════════════════════════════════════════════════════════════

Single lazy connection used by the cross-process primitives:
  - login rate limiting (core/rate_limit.py)
  - irreversible-call double-send guard (core/tool_executor.py)
(the delivery-outbox claim is PostgreSQL FOR UPDATE SKIP LOCKED — no Redis
needed there).

Graceful degradation is the contract: `get_redis()` returns None when
`redis.url` is unconfigured, the redis package is missing, or the server is
unreachable — every caller MUST keep its in-process fallback. A Redis
outage therefore degrades to per-worker behavior, never to a crash.

Config:  redis.url = redis://localhost:6379/0   (unset → disabled)
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger("agent.redis")

_lock = threading.Lock()
_client = None
_failed_until = 0.0
_RETRY_S = 30.0


def get_redis():
    """Return a connected redis client, or None (degraded)."""
    global _client, _failed_until
    if _client is not None:
        return _client
    if time.time() < _failed_until:
        return None
    with _lock:
        if _client is not None:
            return _client
        try:
            from core.infra_config import redis_url
            url = redis_url()
        except Exception:
            url = ""
        if not url:
            return None
        try:
            import redis
            client = redis.Redis.from_url(
                url, socket_timeout=2, socket_connect_timeout=2,
                decode_responses=True)
            client.ping()
            _client = client
            logger.info("redis.connected: %s", url.split("@")[-1])
            return _client
        except Exception as e:
            _failed_until = time.time() + _RETRY_S
            logger.warning("redis.unavailable (%s) — using in-process fallbacks", e)
            return None


def reset_for_tests():
    """Drop the cached client/circuit state (test helper)."""
    global _client, _failed_until
    with _lock:
        _client = None
        _failed_until = 0.0


def _channel(name: str) -> str:
    """Pub/sub channels are server-wide in Redis (not per DB index), so two
    deployments sharing one Redis would hear each other's signals. With
    AG_DATA_NAMESPACE set, channels become ag:<ns>:<name>."""
    ns = os.environ.get("AG_DATA_NAMESPACE", "").strip().lower()
    return f"ag:{ns}:{name}" if ns else f"ag:{name}"


CONFIG_RELOAD_CHANNEL = _channel("config:reload")


def publish_config_reload(changed_key: str = ""):
    """Publish a config-reload signal to all subscribed processes.

    Args:
        changed_key: The config key that changed (informational, for logging).
    """
    client = get_redis()
    if client is None:
        return
    try:
        client.publish(CONFIG_RELOAD_CHANNEL, changed_key or "*")
        logger.debug("redis.config_reload.published key=%s", changed_key or "*")
    except Exception as e:
        logger.warning("redis.config_reload.publish_failed: %s", e)


def subscribe_config_reload(callback):
    """Subscribe to config-reload signals in a background thread.

    The callback receives (changed_key: str) and should reload configs.
    Runs as a daemon thread — dies with the process.

    Args:
        callback: Function(changed_key: str) invoked on each reload signal.
    """
    _subscribe_channel(CONFIG_RELOAD_CHANNEL, callback, "config-reload-sub")


SESSION_INVALIDATE_CHANNEL = _channel("agent:session:invalidate")


def publish_session_invalidate(scope: str):
    """Publish a session-invalidation signal to all subscribed workers.

    Args:
        scope: A scope token — "persona:<id>", "app:<id>",
               "integration:<id>", or "*" for a full flush.
    """
    client = get_redis()
    if client is None:
        return
    try:
        client.publish(SESSION_INVALIDATE_CHANNEL, scope or "*")
        logger.debug("redis.session_invalidate.published scope=%s", scope or "*")
    except Exception as e:
        logger.warning("redis.session_invalidate.publish_failed: %s", e)


def subscribe_session_invalidate(callback):
    """Subscribe to session-invalidation signals in a background thread.

    The callback receives (scope: str). Runs as a daemon thread.

    Args:
        callback: Function(scope: str) invoked on each invalidation signal.
    """
    _subscribe_channel(
        SESSION_INVALIDATE_CHANNEL, callback, "session-invalidate-sub")


def _subscribe_channel(channel: str, callback, thread_name: str):
    """Shared pub/sub subscriber loop on a daemon thread.

    Each subscription owns a separate BLOCKING connection. NOTE: socket_timeout
    must be None (block forever) — a 0 timeout makes the socket non-blocking, so
    listen() dies instantly with EAGAIN ("Resource temporarily unavailable").
    Health-check the link with socket_keepalive instead of a read timeout.

    The thread keeps (re)connecting with backoff rather than checking Redis once
    and giving up. This is essential on prod: subscribers start during app
    boot, when Redis may not be reachable for a split second — the old
    check-once-then-skip behaviour left every process permanently unsubscribed,
    so cross-process invalidations were published but never delivered. Retrying
    also survives a Redis restart mid-flight.
    """

    def _listener():
        backoff = 1.0
        waited_for_config = 0.0
        while True:
            try:
                import redis as redis_lib
                from core.infra_config import redis_url
                url = redis_url()
                if not url:
                    if waited_for_config < 120.0:
                        waited_for_config += backoff
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 15.0)
                        continue
                    logger.info(
                        "redis.subscriber(%s): redis.url still unset after %.0fs — "
                        "assuming Redis is not configured; polling slowly",
                        channel, waited_for_config)
                    time.sleep(30.0)
                    continue
                waited_for_config = 0.0
                sub_client = redis_lib.Redis.from_url(
                    url, socket_timeout=None, socket_keepalive=True,
                    decode_responses=True)
                sub_client.ping()
                pubsub = sub_client.pubsub()
                pubsub.subscribe(channel)
                logger.info("redis.subscriber: listening on %s", channel)
                backoff = 1.0
                for message in pubsub.listen():
                    if message["type"] == "message":
                        data = message.get("data", "")
                        logger.info("redis.subscriber(%s).received data=%s",
                                    channel, data)
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error("redis.subscriber(%s).callback_error: %s",
                                         channel, e)
                logger.warning("redis.subscriber(%s): stream ended — reconnecting", channel)
            except Exception as e:
                logger.warning(
                    "redis.subscriber(%s): connect/listen failed (%s) — retry in %.0fs",
                    channel, e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    t = threading.Thread(target=_listener, name=thread_name, daemon=True)
    t.start()
