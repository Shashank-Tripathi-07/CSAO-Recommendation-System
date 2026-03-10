import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

class Monitoring:
    def __init__(self):
        self.logger = logging.getLogger("csao-monitoring")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_metrics(self, metrics: Dict[str, Any], context: str = "general", metadata: Dict[str, Any] = None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "context": context,
            "metrics": metrics,
            "metadata": metadata or {}
        }
        self.logger.info(json.dumps(log_entry))

    def log_error(self, message: str, context: str = "general", exc: Optional[Exception] = None, metadata: Dict[str, Any] = None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR",
            "context": context,
            "message": message,
            "metadata": metadata or {}
        }
        if exc:
            log_entry["exception"] = {
                "type": type(exc).__name__,
                "details": str(exc),
                "traceback": traceback.format_exc()
            }
        self.logger.error(json.dumps(log_entry))

monitor = Monitoring()
