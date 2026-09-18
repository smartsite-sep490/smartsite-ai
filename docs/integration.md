# Tích hợp SmartSite Backend

Nguồn chuẩn: thư mục `contracts/` trong repository `smartsite`.

Phiên bản event contract đang dùng: **chưa phát hành**. Không có event endpoint/path/schema mặc định được coi là đã chốt.

## Contract foundation đang chạy

Service mặc định ở `127.0.0.1:8000`; trong Compose, các container cùng mạng dùng `http://ai:8000` nếu service được đặt tên `ai`. Backend chưa có consumer hoặc job inference kết nối service này.

- `GET /health/live`: 200, `{"status":"ok","service":"smartsite-ai"}`.
- `GET /health/ready`: 200 sau startup, `{"status":"ready","service":"smartsite-ai","scope":"api","inference_ready":false}`. Ngoài lifecycle trả 503 với `status: "not_ready"`.
- `GET /v1/capabilities`: `api_version: "v1"`, `inference_ready: false`, `capabilities` gồm camera, detector, zone, identity, openai. Mỗi capability có `status: "not_configured"`, `provider` và `reason`. Các ghi chú không chứa secret hoặc model path.

Không dùng API readiness để bật nghiệp vụ PPE/Zone. Service chưa phát event, chưa có camera scheduler hoặc kết nối database/Backend/OpenAI. OpenAPI development mô tả chính các HTTP endpoint hiện có.

Khi triển khai contract đầu tiên, ghi version và commit nguồn, thêm schema/example tương ứng có nguồn gốc rõ, chạy kiểm tra trước khi gửi event. Không sửa một bản sao schema riêng rồi coi hai repo đã đồng bộ.

## Thiết kế dự kiến

- Backend cấp cấu hình camera/Zone theo phạm vi AI được phép.
- AI quan sát video, tạo ID sự kiện ổn định và tham chiếu bằng chứng.
- Backend xác thực producer, kiểm tra payload và chống trùng khi AI retry.
- Backend quyết định quyền vào theo identity, Zone và thời gian; AI không truy cập trực tiếp database nghiệp vụ.
- Mất quyền truy cập hoặc không định danh được phải có trạng thái riêng.

HTTP/event transport, xác thực, timeout/retry và cache chính sách còn cần chốt ở contract.

## Release

AI có version/image riêng. Mỗi release ghi model version, cấu hình có ảnh hưởng, phiên bản contract và Backend đã kiểm thử cùng. Thay model không mặc nhiên đồng nghĩa API thay đổi; cần chạy lại đo chất lượng và ca nghiệm thu.
