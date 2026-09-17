# Tích hợp SmartSite Backend

Nguồn chuẩn: thư mục `contracts/` trong repository `smartsite`.

Phiên bản contract đang dùng: **chưa phát hành**. Không có endpoint/path/schema mặc định được coi là đã chốt.

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
