# SmartSite AI

Service AI độc lập cho SmartSite MF05 (PPE) và MF06 (Zone theo quyền từng người).

**Trạng thái:** FastAPI foundation chạy được với health/capability endpoints. Chưa có camera ingestion, inference, nhận diện PPE/identity, event producer hoặc OpenAI adapter. API readiness không chứng minh inference hoạt động.

Hướng pipeline đã chọn: **FastAPI + YOLO26 + Supervision**. **InsightFace** là ứng viên định danh cần đánh giá; **OpenAI** dùng bổ sung phân tích bằng chứng trong milestone sau. Backend sở hữu quyền nghiệp vụ và quy trình sự cố; AI không truy cập database nghiệp vụ. Track ID không phải Worker ID.

## Chạy local

Cài [uv](https://docs.astral.sh/uv/getting-started/installation/) và dùng Python **3.12.13**. Manifest và `uv.lock` khóa dependency; CI/Docker dùng uv **0.11.6**.

```sh
uv python install 3.12.13
uv sync --frozen
uv run --frozen smartsite-ai
```

Mặc định nghe ở `127.0.0.1:8000`. Không cần CUDA, model weights, camera hoặc OpenAI key. `.env.example` ghi các biến tùy chọn; có thể sao chép thành `.env` bằng `Copy-Item .env.example .env` trên PowerShell hoặc `cp .env.example .env` trên Unix. Environment overrides `.env`.

| Biến | Mặc định | Giá trị |
| --- | --- | --- |
| `SMARTSITE_AI_ENVIRONMENT` | `development` | `development`, `test`, `production` |
| `SMARTSITE_AI_HOST` | `127.0.0.1` | Địa chỉ IPv4/IPv6 hợp lệ |
| `SMARTSITE_AI_PORT` | `8000` | Số nguyên 1–65535 |
| `SMARTSITE_AI_LOG_LEVEL` | `info` | `critical`, `error`, `warning`, `info`, `debug`, `trace` |

Cấu hình không hợp lệ dừng startup. Service chưa đọc camera credentials/model paths/API keys; đặt những biến đó không bật tính năng.

## HTTP

| Endpoint | Ý nghĩa |
| --- | --- |
| `GET /health/live` | 200 khi process phục vụ HTTP: `{"status":"ok","service":"smartsite-ai"}` |
| `GET /health/ready` | 200 sau ASGI startup, 503 ngoài lifecycle; `scope: "api"`, `inference_ready: false` |
| `GET /v1/capabilities` | Camera, detector, Zone, identity, OpenAI đều `status: "not_configured"` với lý do |
| `GET /docs`, `/openapi.json` | OpenAPI từ route thực, chỉ bật ngoài production |

```sh
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/v1/capabilities
```

Chưa có endpoint xử lý hình ảnh hoặc endpoint nghiệp vụ. Không coi shell này là deployment production đã hoàn tất auth, storage, observability và event delivery.

## Kiểm tra

```sh
uv lock --check
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest
uv build
```

Tests chạy với fake configuration, không gọi API trả phí hoặc tải model. CI chạy các lệnh trên, kiểm tra wheel import và build/smoke image. Starlette 1.6 [TestClient](https://github.com/Kludex/starlette/blob/1.6.0/starlette/testclient.py) ưu tiên `httpx2` và cảnh báo khi dùng `httpx`, nên dev dependency khóa `httpx2==2.13.0`. Một filter hẹp trong pytest bỏ cảnh báo AnyIO alias cũ do chính TestClient này phát ra; các warning khác vẫn là lỗi.

Phiên bản core: FastAPI **0.141.1**, Uvicorn **0.53.0**, pydantic-settings **2.15.0**. Optional `vision` khóa Ultralytics **8.4.155** và Supervision **0.30.4**, nhưng chưa cài/chạy trong kiểm thử foundation. Lockfile chứa dependency bắc cầu; không chạy `--all-extras` cho setup API thường ngày.

## Docker

```sh
docker build -t smartsite-ai:dev .
docker run --rm --name smartsite-ai -p 127.0.0.1:8000:8000 smartsite-ai:dev
```

Image CPU chỉ cài core bằng `uv sync --frozen --no-dev --no-editable`, chạy UID/GID 10001, bind `0.0.0.0` bên trong container và kiểm tra `/health/ready`. Build context chỉ cho phép source/manifest/lock/README, không chứa `.env`, datasets hoặc model artifacts. Compose tích hợp nằm ở repo `smartsite/infra`; clone hai repo cạnh nhau để dùng profile AI.

## GPU và bước inference tiếp theo

RTX 4060, **1–3 camera**, là mục tiêu thử khả thi; chưa có số FPS, latency hay độ chính xác được chứng minh. Xem [kế hoạch kiểm chứng GPU](docs/feasibility.md) trước khi bật extra hoặc tải weights. `uv sync --frozen --extra vision` chỉ dành cho môi trường thử riêng sau khi chọn PyTorch/CUDA phù hợp; extra mặc định chưa phải cấu hình GPU đã kiểm chứng và có thể tải package lớn. Chạy `uv sync --frozen` trở lại để giữ môi trường core.

- [Tích hợp Backend](docs/integration.md): health contract hiện có, event contract chưa phát hành.
- [Model artifacts](models/README.md): version, checksum và dataset đánh giá ngoài Git.
- [CONTRIBUTING.md](CONTRIBUTING.md): quy trình review, giữ nhánh và phạm vi dữ liệu.
