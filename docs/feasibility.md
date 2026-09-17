# Thử khả thi MF05 / MF06

Tài liệu đề xuất để đo trước khi hứa về độ chính xác hoặc tốc độ.

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
