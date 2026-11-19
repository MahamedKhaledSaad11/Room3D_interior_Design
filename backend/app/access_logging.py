from __future__ import annotations

import logging
import re

_CREDENTIAL_QUERY = re.compile(r"([?&](?:token|access_token)=)[^&\s]*", re.IGNORECASE)


class RedactAccessCredentials(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_CREDENTIAL_QUERY.sub(r"\1[redacted]", value) if isinstance(value, str) else value
                                for value in record.args)
        return True
