import time
import logging
from prometheus_client import Counter, Histogram

REQUESTS = Counter("app_requests_total", "Total API requests", ["route", "method", "code"])
LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["route", "method"])

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def observe_request(route: str, method: str):
    start = time.time()
    def _finalize(status_code: int):
        LATENCY.labels(route=route, method=method).observe(time.time() - start)
        REQUESTS.labels(route=route, method=method, code=str(status_code)).inc()
    return _finalize
