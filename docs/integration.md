# Tích hợp SmartSite Backend

Nguồn chuẩn: `contracts/schemas/v1/technical-observation-event.json` tại commit bất biến ghi trong `contracts/metadata.json` của repository `smartsite`.

Phiên bản event contract đang dùng: **1.0.0**. Schema và golden vectors được vendor nguyên bytes; CI tải đúng source commit và fail nếu có drift.

## Contract foundation đang chạy

Service mặc định ở `127.0.0.1:8000`; trong Compose, các container cùng mạng dùng `http://ai:8000`. Backend consumer hiện là `POST http://backend:3000/api/v1/integrations/ai/events`, xác thực bằng Bearer service token.

- `GET /health/live`: 200, `{"status":"ok","service":"smartsite-ai"}`.
- `GET /health/ready`: 200 sau startup, `{"status":"ready","service":"smartsite-ai","scope":"api","inference_ready":false}`. Ngoài lifecycle trả 503 với `status: "not_ready"`.
- `GET /v1/capabilities`: `api_version: "v1"`, `inference_ready: false`, `capabilities` gồm camera, detector, zone, identity, openai. Mỗi capability có `status: "not_configured"`, `provider` và `reason`. Các ghi chú không chứa secret hoặc model path.

Không dùng API readiness để bật nghiệp vụ PPE/Zone. Contract models, canonical hash và Backend client đã có; camera scheduler/producer loop, inference và OpenAI adapter chưa có. OpenAPI development mô tả chính các HTTP endpoint hiện có.

Mọi thay đổi contract phải bắt đầu ở repo `smartsite`, commit schema trước, sau đó vendor bytes và cập nhật source SHA/checksums ở repo này. Không sửa schema vendored trước.

## Thiết kế dự kiến

- Backend cấp cấu hình camera/Zone theo phạm vi AI được phép.
- AI quan sát video, tạo ID sự kiện ổn định và tham chiếu bằng chứng.
- Backend xác thực producer, kiểm tra payload và chống trùng khi AI retry.
- Backend quyết định quyền vào theo identity, Zone và thời gian; AI không truy cập trực tiếp database nghiệp vụ.
- Mất quyền truy cập hoặc không định danh được phải có trạng thái riêng.

Client có timeout riêng cho connect/read/write/pool; retry tối đa ba lần chỉ với lỗi transport, 408, 429 và 5xx. 4xx nghiệp vụ không retry. Mọi request dict được Pydantic validate trước khi gửi; acknowledgement phải có status hợp lệ, `alertIds` và cùng `eventId`.

## Release

AI có version/image riêng. Mỗi release ghi model version, cấu hình có ảnh hưởng, phiên bản contract và Backend đã kiểm thử cùng. Thay model không mặc nhiên đồng nghĩa API thay đổi; cần chạy lại đo chất lượng và ca nghiệm thu.
