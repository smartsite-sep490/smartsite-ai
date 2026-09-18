# Cách nhóm cùng làm việc

Cả nhóm làm xuyên Web, Backend, Mobile và AI. Chọn người thực hiện và người review theo đầu việc, không chia cố định theo tầng.

1. Tạo issue có MF/FR/UC liên quan, phạm vi và tiêu chí nghiệm thu.
2. Tạo nhánh ngắn từ `main`, ví dụ `feat/mf05-alert-list`, `fix/mf06-unknown-identity` hoặc `docs/mf05-use-case`. Nhánh do Codex tạo dùng tiền tố `codex/`.
3. Làm code và cập nhật tài liệu/contract trong cùng PR. Một PR có thể chạm nhiều ứng dụng nếu phục vụ cùng một hành vi.
4. Ghi rõ đã kiểm thử gì. Nhờ một thành viên khác review; xử lý góp ý, squash merge và giữ lại nhánh đã merge. GitHub đã tắt tự xóa nhánh; chỉ xóa thủ công khi nhóm chủ động quyết định.
5. Khi thay đổi cả hai repo, liên kết PR và ghi phiên bản contract tương thích. Không giả định hai PR được merge hoặc deploy đồng thời.

`main` là nhánh tích hợp chung; ban đầu không cần thêm nhánh `develop` dài hạn. Không force-push lên nhánh dùng chung.

## CI hiện tại và giới hạn

Workflow kiểm tra whitespace, uv lock/install, Ruff lint/format, pytest config/HTTP, Python distribution và Docker build/smoke. Chạy các lệnh trong README trước khi gửi PR. Chưa có kiểm tra model, camera, GPU hoặc nghiệp vụ PPE/Zone; khi thêm event contract máy đọc được, thêm kiểm tra producer/consumer.

Với Organization Free và repo private, quy tắc review ở trên là quy ước nhóm; không khẳng định GitHub đang cưỡng chế protected branch. Xem [tài liệu GitHub về protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

## Trước khi commit

Dùng `git diff --cached` để xem nội dung sắp đưa lên Git. Chỉ dùng dữ liệu giả cho fixture. File cấu hình mẫu phải chứa giá trị giả; secrets, video camera, ảnh/mẫu khuôn mặt và trọng số model lưu ngoài Git.

Không chọn người review hay đưa tên thành viên vào CODEOWNERS khi chưa có phân công của nhóm.
