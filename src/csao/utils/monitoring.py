import json
import logging
from datetime import datetime
from typing import Dict, Any

class Monitoring:
    def __init__(self):
        self.logger = logging.getLogger("csao-monitoring")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)

    def log_metrics(self, metrics: Dict[str, Any], context: str = "general", metadata: Dict[str, Any] = None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "metrics": metrics,
            "metadata": metadata or {}
        }
        self.logger.info(json.dumps(log_entry))

monitor = Monitoring()
