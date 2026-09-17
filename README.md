# SmartSite AI

Repository độc lập cho AI camera của SmartSite, tập trung MF05 (PPE) và MF06 (Zone theo quyền từng người).

**Trạng thái:** khung repository; chưa có inference server hoặc model đã kiểm chứng. Tài liệu hiện đề xuất Python/FastAPI và các model nhận diện; chưa chốt framework/model cuối cùng.

## Cấu trúc

- `src/`: code runtime AI khi bắt đầu triển khai.
- `tests/`: kiểm tra đơn vị và contract; chỉ dùng fixture giả hoặc dữ liệu được phép.
- `docs/`: kế hoạch khả thi và cách tích hợp Backend.
- `models/`: hướng dẫn quản lý model; trọng số lưu ngoài Git.
- `.github/`: mẫu issue/PR và kiểm tra repository ban đầu.

## Ranh giới

AI tạo quan sát kỹ thuật và bằng chứng. Backend quản lý quyền nghiệp vụ và quy trình cảnh báo/sự cố. Track ID không phải Worker ID; unknown identity không đồng nghĩa allowed hoặc confirmed violation.

Contract chuẩn nằm ở `smartsite/contracts`; xem [tích hợp](docs/integration.md). Có thể clone repo này cạnh `smartsite`; không phải chép source AI vào monorepo.

## Bắt đầu

1. Đọc [kế hoạch thử khả thi](docs/feasibility.md).
2. Chốt runtime, model và contract cùng nhóm.
3. Thêm package manifest/lockfile, lệnh chạy, cấu hình mẫu và CI test trước khi tích hợp camera thật.
4. Thêm image riêng khi có entrypoint/healthcheck; Compose tích hợp chung nằm ở repo `smartsite/infra`.

Chưa có lệnh chạy ứng dụng tại thời điểm này. Xem [CONTRIBUTING.md](CONTRIBUTING.md) cho quy trình code/doc/review.
