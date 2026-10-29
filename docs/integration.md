# Tích hợp SmartSite Backend

Nguồn chuẩn: `contracts/schemas/v1/technical-observation-event.json` tại commit bất biến ghi trong `contracts/metadata.json` của repository `smartsite`.

Phiên bản event contract đang dùng: **1.0.0**. Schema và golden vectors được vendor nguyên bytes; CI tải đúng source commit và fail nếu có drift.

Schema PPE mở rộng `1.1.0` được vendor riêng từ Platform commit bất biến trong
`metadata.expandedObservationEvent`; provenance v1 không đổi. Python event
reject GLOVES/BOOTS/GOGGLES khi version là 1.0.0. Orchestrator phải chọn rõ
consumer version 1.1.0 cho `PpePipeline.for_model_profile(native10)`; không chọn
version thì khởi tạo fail-closed. Runtime assembly mặc định vẫn dùng legacy.
Association và temporal confirmation mở rộng vẫn bỏ evidence mâu thuẫn hoặc
không có chủ thể duy nhất; không tạo PPE missing từ nhãn model không hỗ trợ.

`load_artifact_spec(..., experimental_profile="native-ppe-10")` là opt-in đọc
taxonomy thử nghiệm 10 lớp. Mặc định loader và worker vẫn dùng đúng 5 lớp cũ;
opt-in không bật serving, không đổi wire contract hoặc cấp model acceptance.
Profile có Gloves/Boots/Goggles nhưng không có NO-Vest/Harness/Hook. ID lớp 4
là Gloves trong native10, NO-Vest trong baseline; không dùng integer ID để suy
PPE item khi thiếu profile. `inference/ppe_profiles.py` ghi evidence families
được khai báo, không chứng minh độ chính xác, đủ quan sát hay deployment readiness.

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

## Local diagnostic camera preview

The authenticated `/ws/realtime` demo endpoint sends `previewVersion: 1` messages containing
the exact inference frame as `imageDataUrl` (JPEG), its detections, and the configured restricted
`zonePolygons`. Each message includes `cameraExternalId`, `sessionId`, decimal-string
`sequenceNumber`, `capturedAt`, and encoded image `width`/`height`. JPEG output is bounded to
1280 pixels per dimension and 1 MiB before Base64 encoding. Normalized boxes and polygons use
the same image coordinate space. Empty detection frames still carry pixels and source identity.

Clients must display the image and its boxes together, bound image decoding, and discard stale
work after disconnect/session changes. A separately playing MP4 cannot be used as the background
for realtime boxes. These transient diagnostic pixels do not replace authenticated retained
evidence or change `TechnicalObservationEvent` v1.0.0. The controlled laptop/Tapo profiles remain
exclusive to the durable worker; shared live-camera preview fan-out is not implemented here.
Model/artifact initialization runs in a background thread after the client connects, so it does
not occupy the API event loop. Cancellation preserves ownership of the loading task and closes
a model that finishes loading after cancellation, including repeated cancellation requests.
Provider construction serializes its process-wide offline guards with a threading lock:
a replacement connection cannot restore download/auto-install defaults while a cancelled
connection's native model loader is still constructing a checkpoint.

Cancelling a YOLO detection waits for its already-started native prediction to finish before
propagating cancellation. This preserves the shared inference lane and prevents single-camera,
multicamera, or preview cleanup from closing the runner while prediction is active. Repeated
cancellation does not abandon that work; a late provider error is consumed while cancellation
remains the caller outcome. This is graceful ownership, not a hard shutdown timeout: a permanently
hung native provider still requires process-level recovery.

## Release

AI có version/image riêng. Mỗi release ghi model version, cấu hình có ảnh hưởng, phiên bản contract và Backend đã kiểm thử cùng. Thay model không mặc nhiên đồng nghĩa API thay đổi; cần chạy lại đo chất lượng và ca nghiệm thu.

### PPE class binding before a model change

The official worker artifact loader currently requires the exact five-class
baseline map (`Person`, `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`)
and checks it against checkpoint metadata. A candidate with different native
class names or IDs cannot be installed by changing its spec labels to the
baseline: the spec must describe the actual checkpoint.

At the provider-neutral pipeline boundary, `PpePipeline` also recognizes the
explicit negative aliases `no-helmet` and `no-vest` as `HARD_HAT:MISSING` and
`SAFETY_VEST:MISSING`. This permits correctly bound diagnostic batches; it does
not relax artifact loading or approve candidate deployment. Positive and
negative evidence for the same item, or PPE fitting multiple people, remains
unknown. The default legacy profile ignores gloves, boots and harness rather
than mapping them to a vest. Expanding the emitted PPE scope requires the canonical
Backend contract update and paired compatibility checks described above.

The experimental `native-ppe-10` profile preserves actual checkpoint IDs and
adds Gloves/Boots/Goggles association. Camera runtime manifests must opt in with
`experimentalModelProfile: "native-ppe-10"` and
`observationSchemaVersion: "1.1.0"`; omitted fields retain the legacy behavior.
Realtime uses the corresponding explicit settings in `.env.example`. Selection
can use `realtime_artifact_spec_path` to retain the artifact's image size,
confidence and NMS settings instead of the legacy raw-path defaults. A spec
cannot be combined with raw model/class-map paths. An explicitly configured
`realtime_device` overrides only the artifact device, allowing CPU deployment;
the evaluated preprocessing remains unchanged. Spec loading checks YOLO11
checkpoint metadata for both the legacy five-class and native10 profiles.
Selection
does not prove that a deployed Backend accepts v1.1: operators must verify the
consumer deployment before starting delivery. Configuration fails before opening
resources if the expanded profile is paired with v1.0.

Native10 has no NO-Safety Vest, Harness, hook or protective-clothing classes.
Missing evidence remains UNKNOWN; it cannot supply missing-vest coverage from
the legacy model. The v1.1 schema is currently pinned to the published Platform
feature checkpoint for paired development. Finalize provenance to its immutable
main merge SHA before either repository's main/release integration. Do not treat
the feature pin, passing compatibility tests or artifact selection as acceptance.
