# Thử khả thi MF05 / MF06

Tài liệu đề xuất để đo trước khi hứa về độ chính xác hoặc tốc độ.

## Baseline và thử RTX 4060

Foundation hiện chỉ chạy HTTP. Hướng đã chốt là FastAPI + YOLO26 + Supervision; InsightFace vẫn là ứng viên identity. OpenAI bổ sung phân tích bằng chứng, không thay detector hay quyết định quyền Zone. Chưa có camera/model/GPU validation, license acceptance, đăng ký identity hoặc API call.

1. Dùng môi trường Python 3.12 riêng, ghi hệ điều hành, NVIDIA driver (`nvidia-smi`), dung lượng VRAM thực và CPU/RAM. Target là RTX 4060 với 1–3 camera, không phải cam kết throughput.
2. Chọn PyTorch/CUDA từ [hướng dẫn PyTorch chính thức](https://pytorch.org/get-started/locally/), ghi index và exact version. Không giả định dependency Torch trên PyPI là bản CUDA phù hợp. Khi chọn GPU build khác, cập nhật cấu hình uv/index và lockfile trong PR riêng.
3. Sau khi kiểm tra môi trường, `uv sync --frozen --extra vision` cài extra đã khóa để thử dependency. Lệnh này có thể tải package lớn và chưa được chạy trong foundation. Kiểm tra Torch bằng `uv run --frozen --extra vision python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`. Không import hay tạo model trong API startup.
4. Chọn model/weights cục bộ với version, checksum, nguồn và điều kiện sử dụng; không dùng lệnh khởi tạo tự tải weights trong startup. YOLO26 pretrained chưa phải PPE model đã được kiểm chứng. [YOLO26](https://docs.ultralytics.com/models/yolo26/) và [Supervision](https://supervision.roboflow.com/) là nguồn tham khảo pipeline.
5. Bắt đầu video đã được phép dùng, rồi tăng 1 → 2 → 3 camera; cố định resolution, sampling FPS, batch size và thời gian chạy. Đo cả decode, detector, tracker, Zone/identity và hàng đợi, không chỉ thời gian forward model. Ghi p50/p95 latency, FPS mỗi camera, dropped frames, VRAM/RAM, nhiệt và reconnect behavior.
6. So sánh chất lượng PPE/identity theo điều kiện bên dưới, đặc biệt unknown/occlusion. Chỉ thêm capability trạng thái sẵn sàng khi worker thực được kiểm tra. API health 200 riêng không cho phép kết luận camera đang hoạt động.

Kết quả foundation: optional vision đã resolve vào lockfile; chưa cài Torch/CUDA/InsightFace, chưa tải weights, chưa có benchmark RTX 4060 hoặc gọi OpenAI. Việc lựa chọn/kiểm tra identity và OpenAI adapter nằm ở milestone tiếp theo.

## MF05

- Chốt loại PPE cần kiểm tra và vùng áp dụng.
- Ghi model, nguồn trọng số, điều kiện sử dụng/license và cấu hình inference.
- Đo trên tình huống sáng/tối, xa/gần, che khuất, nhiều người.
- Phân biệt thiếu PPE với không quan sát được PPE.
- Ghi false positive/false negative, độ trễ, FPS, hardware và giới hạn.
- Chốt giữ tín hiệu bao lâu, cooldown và chống trùng cùng Backend.

## MF06

- Tách bài thử người vào polygon khỏi bài thử định danh người.
- Chọn phương thức gắn identity có thể kiểm chứng ở camera Zone.
- Track ID chỉ là mã tạm trong camera/session, không chứng minh danh tính.
- Demo tối thiểu ba trường hợp: có quyền, bị cấm, chưa xác định.
- Kiểm tra mất kết nối quyền, hết hạn quyền, đổi chính sách và thay camera/session.

## Đầu ra

Báo cáo kết quả có cách tái lập, số mẫu, điều kiện thử và giới hạn; contract event phù hợp kết quả quan sát; quyết định chọn model/hardware có bằng chứng.

Không đưa video camera thật, mẫu mặt hoặc embeddings vào Git. Bản ghi thử dùng dữ liệu được phép và tham chiếu nơi lưu có kiểm soát.
