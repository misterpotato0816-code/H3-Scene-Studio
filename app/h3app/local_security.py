"""Local browser boundary and platform-independent path validation."""
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

from aiohttp import web


def contained_file(root: Path, name: str) -> Path | None:
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        return None
    return target


def output_prefix(value: str) -> str:
    value = str(value).strip().replace("\\", "/")
    if (not value or value.startswith("/") or PureWindowsPath(value).drive
            or any(p in ("", ".", "..") for p in value.split("/"))
            or any(ord(c) < 32 or c in ':*?"<>|' for c in value)):
        raise ValueError("保存先は出力フォルダ内の相対名にしてください（例: H3/APP）。")
    return value


@web.middleware
async def local_boundary(request, handler):
    # Reject DNS rebinding as well as a hostile website calling localhost.
    try:
        target = urlsplit("http://" + request.host)
        valid_host = target.hostname in ("127.0.0.1", "localhost", "::1")
        target_port = target.port or 80
    except ValueError:
        valid_host = False
    if not valid_host:
        raise web.HTTPForbidden(text="ローカルのH3画面から操作してください。")
    origin = request.headers.get("Origin")
    if origin:
        try:
            source = urlsplit(origin)
            same = (source.scheme == "http" and source.hostname == target.hostname
                    and (source.port or 80) == target_port
                    and not source.username and not source.password
                    and not source.path and not source.query and not source.fragment)
        except ValueError:
            same = False
        if not same:
            raise web.HTTPForbidden(text="外部ページからの操作は受け付けません。")
    if request.path.startswith("/api/"):
        if request.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
            raise web.HTTPForbidden(text="H3画面から操作してください。")
        if request.method in ("POST", "PUT", "PATCH"):
            allowed = ("application/json",)
            if request.path == "/api/images":
                allowed += ("multipart/form-data",)
            if request.content_type not in allowed:
                raise web.HTTPUnsupportedMediaType(text="JSON形式で送信してください。")
    response = await handler(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response
