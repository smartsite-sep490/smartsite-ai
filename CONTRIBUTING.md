# Cách nhóm cùng làm việc

Cả nhóm làm xuyên Web, Backend, Mobile và AI. Chọn người thực hiện và người review theo đầu việc, không chia cố định theo tầng.

1. Tạo issue có MF/FR/UC khi áp dụng, phạm vi và tiêu chí nghiệm thu. Công việc dependency, CI, security hoặc docs không được tự bịa mã truy vết.
2. Tạo nhánh ngắn từ `main`, ví dụ `feat/mf05-alert-list`, `fix/mf06-unknown-identity` hoặc `docs/mf05-use-case`. Nhánh do Codex tạo dùng tiền tố `codex/`.
3. Làm code và cập nhật tài liệu/contract trong cùng PR. Một PR có thể chạm nhiều ứng dụng nếu phục vụ cùng một hành vi.
4. Ghi rõ đã kiểm thử gì. Nhờ một thành viên khác review, xử lý góp ý và squash merge. Lịch sử được giữ trong PR; chỉ xóa nhánh thủ công sau khi nhóm xác nhận không còn công việc tiếp nối và worktree liên quan đã sạch.
5. Khi đổi contract hai repo, có thể chuẩn bị hai PR song song nhưng phải merge contract chuẩn vào `smartsite` trước. Sau đó lấy SHA cuối trên `main`, cập nhật schema/provenance trong `smartsite-ai`, chạy paired checks rồi mới merge PR AI. Không giả định hai repo deploy đồng thời; breaking change phải hỗ trợ chuyển tiếp theo version.

`main` là nhánh tích hợp chung; ban đầu không cần thêm nhánh `develop` dài hạn. Không force-push lên nhánh dùng chung.

## CI hiện tại và giới hạn

Workflow kiểm tra whitespace, uv lock/install, Ruff lint/format, pytest config/HTTP, Python distribution và Docker build/smoke. Chạy các lệnh trong README trước khi gửi PR. Chưa có kiểm tra model, camera, GPU hoặc nghiệp vụ PPE/Zone; khi thêm event contract máy đọc được, thêm kiểm tra producer/consumer.

Với Organization Free và repo private, quy tắc review ở trên là quy ước nhóm; không khẳng định GitHub đang cưỡng chế protected branch. Xem [tài liệu GitHub về protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

## Trước khi commit

Dùng `git diff --cached` để xem nội dung sắp đưa lên Git. Chỉ dùng dữ liệu giả cho fixture. File cấu hình mẫu phải chứa giá trị giả; secrets, video camera, ảnh/mẫu khuôn mặt và trọng số model lưu ngoài Git.

Không chọn người review hay đưa tên thành viên vào CODEOWNERS khi chưa có phân công của nhóm.
