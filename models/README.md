# Model artifacts

Do not commit weights. Keep every runtime artifact local and record its source, exact version,
SHA-256, license status, class map, configuration, and evaluation evidence. The selected
implementation baseline for MF05/MF06 is YOLO11s. No validated PPE checkpoint has been selected,
trained, or acquired for SmartSite. RF-DETR Nano/Small and YOLO26s are optional benchmark
challengers. InsightFace remains an identity candidate without a selected model or enrollment
store. Optional vision dependencies do not enable inference at API startup. Ultralytics artifacts
are AGPL-3.0 by default; proprietary or commercial use needs release-license review.

## Local PPE reference inputs — smoke-only

The following two files are permitted local reference inputs for exercising the annotated-video
CLI. They are ignored by Git and must not be redistributed. They do **not** constitute a
validated SmartSite MF05 model, a YOLO11s checkpoint, or a benchmark result.

| Local path | Role | Pinned repository commit | Declared repository license | Download SHA-256 |
| --- | --- | --- | --- | --- |
| `recordings/ppe-reference.mp4` | Local permitted-reference input for the CLI smoke run. Its media provenance is not separately stated. | [Morteza-Asadi-Shalmaiy/PPE-Detection-YOLOv8 @ `884d6d9a9ad1b07b8a2f9330a8c53beda907474d`](https://github.com/Morteza-Asadi-Shalmaiy/PPE-Detection-YOLOv8/commit/884d6d9a9ad1b07b8a2f9330a8c53beda907474d) | MIT | `12e9c8d27f904c18aa834c49de5dc402c12fbdcafb8448443a5610c4c11668a4` |
| `models/ppe-yolov8-reference.pt` | Compatible YOLOv8 PPE checkpoint for the CLI smoke run only. | [Ansarimajid/Construction-PPE-Detection @ `8139436e91aecb109362e13cacfea44a16e08358`](https://github.com/Ansarimajid/Construction-PPE-Detection/commit/8139436e91aecb109362e13cacfea44a16e08358) | MIT | `5c981fd81432236cd6c88fa336697370f110383a62cc967f7759debf3c2b147e` |

The Morteza repository declares MIT, but it does not separately establish the sample media's
provenance. The Ansarimajid repository declares MIT, but the checkpoint's dataset, training
configuration, evaluation metrics, and artifact-license provenance are not established. Treat
`ppe-yolov8-reference.pt` as **evaluation-only**. Do not call it YOLO11s, deploy it, or use it to
make MF05 accuracy, safety, or compliance claims.

Download only these two pinned files into ignored local paths and verify the bytes:

```powershell
New-Item -ItemType Directory -Force recordings, models, .cache, runs | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/PPE-Detection-YOLOv8/884d6d9a9ad1b07b8a2f9330a8c53beda907474d/assets/test-video.mp4' -OutFile recordings/ppe-reference.mp4
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/Ansarimajid/Construction-PPE-Detection/8139436e91aecb109362e13cacfea44a16e08358/Model/ppe.pt' -OutFile models/ppe-yolov8-reference.pt
Get-FileHash -Algorithm SHA256 recordings/ppe-reference.mp4, models/ppe-yolov8-reference.pt
```

The expected hashes are the values in the table. Confirm that the paths stay ignored:

```powershell
git check-ignore -v recordings/ppe-reference.mp4 models/ppe-yolov8-reference.pt
git status --short -- recordings/ppe-reference.mp4 models/ppe-yolov8-reference.pt
```

## Class map for the reference checkpoint

Derive the class map from the downloaded checkpoint each time; do not trust an older README or
external class-list. The command below writes only the ignored local scratch map required by the
CLI:

```powershell
uv run --frozen python -c "import json; from pathlib import Path; from ultralytics import YOLO; names = YOLO('models/ppe-yolov8-reference.pt').names; Path('.cache/ppe-yolov8-reference.class-map.json').write_text(json.dumps({str(key): str(value) for key, value in sorted(names.items())}, indent=2) + '\n', encoding='utf-8')"
```

The checked local reference checkpoint reported this map:

| ID | Class |
| --- | --- |
| 0 | `Hardhat` |
| 1 | `Mask` |
| 2 | `NO-Hardhat` |
| 3 | `NO-Mask` |
| 4 | `NO-Safety Vest` |
| 5 | `Person` |
| 6 | `Safety Cone` |
| 7 | `Safety Vest` |
| 8 | `machinery` |
| 9 | `vehicle` |

The scratch map must remain ignored:

```powershell
git check-ignore -v .cache/ppe-yolov8-reference.class-map.json
```
