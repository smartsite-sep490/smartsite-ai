"""Source setup boundary: host-approved devices, videos and literal RTSP targets."""

from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class SourceRejected(ValueError):
    def __init__(self) -> None:
        super().__init__("SOURCE_NOT_ALLOWED")


class SourceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["WEBCAM", "RTSP", "VIDEO"]
    deviceId: str | None = Field(default=None, min_length=1, max_length=128)
    assetId: str | None = Field(default=None, min_length=1, max_length=128)
    endpoint: SecretStr | None = None
    username: SecretStr | None = None
    password: SecretStr | None = None


@dataclass(frozen=True)
class SourcePolicy:
    webcams: dict[str, int]
    videos: dict[str, Path]
    networks: tuple[str, ...]

    def resolve(self, draft: SourceDraft) -> str:
        fields = draft.model_fields_set - {"kind"}
        if draft.kind == "WEBCAM":
            index = self.webcams.get(draft.deviceId or "")
            if fields != {"deviceId"} or type(index) is not int or not 0 <= index <= 31:
                raise SourceRejected()
            return str(index)
        if draft.kind == "VIDEO":
            path = self.videos.get(draft.assetId or "")
            if (
                fields != {"assetId"}
                or path is None
                or not path.is_absolute()
                or not path.is_file()
            ):
                raise SourceRejected()
            if path.is_symlink() or path.suffix.lower() not in {".mp4", ".mkv", ".avi", ".mov"}:
                raise SourceRejected()
            return str(path)
        if not fields <= {"endpoint", "username", "password"} or draft.endpoint is None:
            raise SourceRejected()
        endpoint = draft.endpoint.get_secret_value()
        if len(endpoint) > 2048 or any(ord(c) <= 32 or ord(c) == 127 for c in endpoint):
            raise SourceRejected()
        try:
            url = urlsplit(endpoint)
            address = ip_address(url.hostname or "")
            allowed = any(address in ip_network(net, strict=True) for net in self.networks)
            if (
                url.scheme not in {"rtsp", "rtsps"}
                or not allowed
                or url.username is not None
                or url.password is not None
                or url.fragment
                or not url.path
                or address.is_multicast
                or address.is_unspecified
                or address.is_link_local
                or (url.port is not None and not 1 <= url.port <= 65535)
            ):
                raise SourceRejected()
        except (ValueError, TypeError):
            raise SourceRejected() from None
        credentials = ""
        if draft.username is not None or draft.password is not None:
            if draft.username is None or draft.password is None:
                raise SourceRejected()
            user, password = draft.username.get_secret_value(), draft.password.get_secret_value()
            if (
                not user
                or len(user) > 256
                or len(password) > 256
                or any(ord(c) < 32 for c in user + password)
            ):
                raise SourceRejected()
            credentials = quote(user, safe="") + ":" + quote(password, safe="") + "@"
        host = "[" + str(address) + "]" if address.version == 6 else str(address)
        port = ":" + str(url.port) if url.port is not None else ""
        return urlunsplit((url.scheme, credentials + host + port, url.path, url.query, ""))
