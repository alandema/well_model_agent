from datetime import datetime
from typing import Any

from langchain.tools import tool


@tool(
    description=(
        "Return today's local date and datetime information, including the "
        "weekday and timezone."
    ),
)
def today_reference() -> dict[str, Any]:
    """Return the current local datetime and useful date components."""
    now = datetime.now().astimezone()
    return {
        "datetime": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.time().isoformat(timespec="seconds"),
        "weekday": now.strftime("%A"),
        "timezone": now.tzname(),
        "utc_offset": now.strftime("%z"),
    }