"""Small integration seam for the deployed V6 pipeline.

Keep this adapter at the point where V6 has both the decoded image and the
final response. It does not transform, filter, or otherwise participate in
prediction construction.
"""

from __future__ import annotations

from typing import Any

from .diagnostic_capture import DiagnosticCapture


def record_v6_diagnostics(
    capture: DiagnosticCapture,
    request: Any,
    decoded_image: Any,
    merged_candidates: Any,
    response: Any,
) -> None:
    """Best-effort diagnostic handoff; never changes the response path."""
    capture.submit(request, decoded_image, merged_candidates, response.annotations)