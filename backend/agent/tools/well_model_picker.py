"""Tool that lists available well models and their descriptions."""

import importlib
import pkgutil
from xml.sax.saxutils import escape

from langchain.tools import tool

from agent import well_models as _pkg


def _discover_models():
    """Import every module in well_models/ and collect model descriptions."""
    models = []
    for module_info in pkgutil.iter_modules(_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{_pkg.__name__}.{module_info.name}")
        for name, obj in vars(module).items():
            if isinstance(obj, type) and hasattr(obj, "description"):
                models.append(
                    {
                        "id": getattr(obj, "id", name),
                        "model_name": name,
                        "description": obj.description,
                    }
                )
    return models


def _to_xml(models):
    """Render the list of models as an XML string."""
    parts = ["<well_models>"]
    for m in models:
        parts.append("  <model>")
        parts.append(f"    <id>{escape(m['id'])}</id>")
        parts.append(f"    <name>{escape(m['model_name'])}</name>")
        parts.append(f"    <description>{escape(m['description'])}</description>")
        parts.append("  </model>")
    parts.append("</well_models>")
    return "\n".join(parts)


@tool(
    description=(
        "List every available well model with its description so you can "
        "choose the best fit for the user's request. Returns an XML "
        "document with one <model> element per available well model."
    ),
)
def well_model_picker() -> str:
    """Return the available well models and their descriptions as XML."""
    return _to_xml(_discover_models())