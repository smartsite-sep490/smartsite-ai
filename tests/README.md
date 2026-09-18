# Kiểm thử AI

Thêm kiểm tra geometry Zone, mapping identity, event contract, retry/chống trùng và dữ liệu không đủ chất lượng khi có code. Fixture commit vào Git chỉ chứa dữ liệu giả hoặc dữ liệu được phép chia sẻ; đánh giá model trên tập dữ liệu lưu riêng.

Hiện chạy `uv run --frozen pytest`: kiểm tra environment validation, API lifecycle/readiness, capability trạng thái chưa cấu hình, không lộ fake credentials và phạm vi OpenAPI. Không cần CUDA, camera, model hoặc API bên ngoài.
