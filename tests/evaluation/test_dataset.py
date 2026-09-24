import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.dataset import (
    MAX_JSONL_LINE_BYTES,
    MAX_MANIFEST_BYTES,
    DatasetValidationError,
    compute_dataset_aggregate_sha256,
    load_evaluation_dataset,
)
from smartsite_ai.evaluation.models import (
    CANONICAL_PPE_CLASSES,
    EvaluationBoundingBox,
    EvaluationFrame,
    EvaluationManifest,
    GroundTruthObject,
    GroundTruthPpeEpisode,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_media(path: Path, content: bytes = b"fake-jpeg-pixels") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return _sha256_bytes(content)


def _create_minimal_dataset(
    tmp_path: Path,
    *,
    custom_manifest_dict: dict | None = None,
    train_rows: list[dict] | None = None,
    val_rows: list[dict] | None = None,
    test_rows: list[dict] | None = None,
    write_media_files: bool = True,
) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir = dataset_dir / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Default frames
    if train_rows is None:
        p1 = images_dir / "train_01.jpg"
        h1 = _write_media(p1, b"train-image-01") if write_media_files else "a" * 64
        train_rows = [
            {
                "frameId": "train_001",
                "mediaPath": "images/train_01.jpg",
                "sha256": h1,
                "width": 1920,
                "height": 1080,
                "annotations": [
                    {
                        "annotationId": "ann_t1_1",
                        "className": "Person",
                        "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9},
                        "personInstanceId": 1,
                        "visibility": 1.0,
                    },
                    {
                        "annotationId": "ann_t1_2",
                        "className": "Hardhat",
                        "boundingBox": {"x1": 0.15, "y1": 0.12, "x2": 0.35, "y2": 0.25},
                        "personInstanceId": 1,
                        "relatedPersonAnnotationId": "ann_t1_1",
                        "visibility": 1.0,
                    },
                ],
            }
        ]

    if val_rows is None:
        p2 = images_dir / "val_01.jpg"
        h2 = _write_media(p2, b"val-image-01") if write_media_files else "b" * 64
        val_rows = [
            {
                "frameId": "val_001",
                "mediaPath": "images/val_01.jpg",
                "sha256": h2,
                "width": 1920,
                "height": 1080,
                "annotations": [
                    {
                        "annotationId": "ann_v1_1",
                        "className": "NO-Hardhat",
                        "boundingBox": {"x1": 0.2, "y1": 0.1, "x2": 0.3, "y2": 0.25},
                        "personInstanceId": 2,
                        "visibility": 1.0,
                    }
                ],
            }
        ]

    if test_rows is None:
        p3 = dataset_dir / "videos" / "test_01.mp4"
        h3 = _write_media(p3, b"test-video-frame-01") if write_media_files else "c" * 64
        test_rows = [
            {
                "frameId": "test_001",
                "mediaPath": "videos/test_01.mp4",
                "sha256": h3,
                "width": 1920,
                "height": 1080,
                "frameIndex": 0,
                "videoTimeSeconds": 0.0,
                "annotations": [
                    {
                        "annotationId": "ann_test1_person_3",
                        "className": "Person",
                        "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.45, "y2": 0.95},
                        "personInstanceId": 3,
                        "visibility": 0.9,
                        "observablePpeItems": ["HARD_HAT", "SAFETY_VEST"],
                    },
                    {
                        "annotationId": "ann_test1_1",
                        "className": "Safety Vest",
                        "boundingBox": {"x1": 0.2, "y1": 0.3, "x2": 0.4, "y2": 0.6},
                        "personInstanceId": 3,
                        "relatedPersonAnnotationId": "ann_test1_person_3",
                        "visibility": 0.9,
                    },
                    {
                        "annotationId": "ann_test1_person_4",
                        "className": "Person",
                        "boundingBox": {"x1": 0.45, "y1": 0.1, "x2": 0.75, "y2": 0.95},
                        "personInstanceId": 4,
                        "visibility": 1.0,
                        "observablePpeItems": ["HARD_HAT", "SAFETY_VEST"],
                    },
                    {
                        "annotationId": "ann_test1_2",
                        "className": "NO-Safety Vest",
                        "boundingBox": {"x1": 0.5, "y1": 0.3, "x2": 0.7, "y2": 0.6},
                        "personInstanceId": 4,
                        "relatedPersonAnnotationId": "ann_test1_person_4",
                        "visibility": 1.0,
                    },
                ],
            }
        ]

    if write_media_files:
        for rows in (train_rows, val_rows, test_rows):
            for r in rows:
                m_path = r.get("mediaPath")
                if m_path and not Path(m_path).is_absolute() and ".." not in m_path:
                    target_file = dataset_dir / m_path
                    if not target_file.exists():
                        declared_hash = r.get("sha256", "")
                        _write_media(
                            target_file, f"bytes-for-declared-sha-{declared_hash}".encode()
                        )
                    r["sha256"] = hashlib.sha256(target_file.read_bytes()).hexdigest()

    (indexes_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n", encoding="utf-8"
    )
    (indexes_dir / "validation.jsonl").write_text(
        "\n".join(json.dumps(r) for r in val_rows) + "\n", encoding="utf-8"
    )
    (indexes_dir / "test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in test_rows) + "\n", encoding="utf-8"
    )

    parsed_splits = {
        "train": tuple(EvaluationFrame.model_validate(r) for r in train_rows),
        "validation": tuple(EvaluationFrame.model_validate(r) for r in val_rows),
        "test": tuple(EvaluationFrame.model_validate(r) for r in test_rows),
    }
    expected_aggregate_sha256 = compute_dataset_aggregate_sha256(parsed_splits)

    manifest_dict = {
        "schemaVersion": "1.0.0",
        "datasetId": "workersafety25",
        "datasetVersion": "1",
        "sourceUrl": "https://universe.roboflow.com/armor/workersafety25",
        "license": "CC BY 4.0",
        "aggregateSha256": expected_aggregate_sha256,
        "classMap": {
            "0": "Person",
            "1": "Hardhat",
            "2": "NO-Hardhat",
            "3": "Safety Vest",
            "4": "NO-Safety Vest",
        },
        "splits": {
            "train": "indexes/train.jsonl",
            "validation": "indexes/validation.jsonl",
            "test": "indexes/test.jsonl",
        },
    }
    if custom_manifest_dict:
        manifest_dict.update(custom_manifest_dict)

    manifest_file = dataset_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    return manifest_file


def test_canonical_ppe_classes_defined() -> None:
    assert CANONICAL_PPE_CLASSES == (
        "Person",
        "Hardhat",
        "NO-Hardhat",
        "Safety Vest",
        "NO-Safety Vest",
    )


def test_valid_five_class_dataset_loads_successfully(tmp_path: Path) -> None:
    manifest_path = _create_minimal_dataset(tmp_path)
    dataset = load_evaluation_dataset(manifest_path)

    assert dataset.manifest.dataset_id == "workersafety25"
    assert dataset.manifest.dataset_version == "1"
    assert set(dataset.splits.keys()) == {"train", "validation", "test"}

    train_frames = dataset.get_split("train")
    assert len(train_frames) == 1
    assert train_frames[0].frame_id == "train_001"
    assert train_frames[0].frame_index is None
    assert train_frames[0].video_time_seconds is None
    assert len(train_frames[0].annotations) == 2

    test_frames = dataset.get_split("test")
    assert len(test_frames) == 1
    assert test_frames[0].frame_index == 0
    assert test_frames[0].video_time_seconds == 0.0

    all_frames = dataset.all_frames()
    assert len(all_frames) == 3


def test_rejects_unknown_classes() -> None:
    invalid_rows = [
        {
            "frameId": "train_001",
            "mediaPath": "images/train_01.jpg",
            "sha256": "a" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "ann_bad_1",
                    "className": "Forklift",
                    "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5},
                    "visibility": 1.0,
                }
            ],
        }
    ]
    with pytest.raises(ValidationError, match="className"):
        EvaluationFrame.model_validate(invalid_rows[0])


def test_rejects_duplicate_frame_ids(tmp_path: Path) -> None:
    dup_rows = [
        {
            "frameId": "frame_dup",
            "mediaPath": "images/f1.jpg",
            "sha256": "1" * 64,
            "width": 640,
            "height": 480,
            "annotations": [],
        },
        {
            "frameId": "frame_dup",
            "mediaPath": "images/f2.jpg",
            "sha256": "2" * 64,
            "width": 640,
            "height": 480,
            "annotations": [],
        },
    ]
    manifest_path = _create_minimal_dataset(tmp_path, train_rows=dup_rows, write_media_files=True)
    with pytest.raises(DatasetValidationError, match="Duplicate frame ID 'frame_dup'"):
        load_evaluation_dataset(manifest_path)


def test_rejects_duplicate_annotation_ids(tmp_path: Path) -> None:
    dup_ann_rows = [
        {
            "frameId": "frame_001",
            "mediaPath": "images/f1.jpg",
            "sha256": "1" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "dup_ann_1",
                    "className": "Person",
                    "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9},
                },
                {
                    "annotationId": "dup_ann_1",
                    "className": "Hardhat",
                    "boundingBox": {"x1": 0.2, "y1": 0.1, "x2": 0.3, "y2": 0.3},
                },
            ],
        }
    ]
    manifest_path = _create_minimal_dataset(
        tmp_path, train_rows=dup_ann_rows, write_media_files=True
    )
    with pytest.raises(DatasetValidationError, match="Duplicate annotation ID 'dup_ann_1'"):
        load_evaluation_dataset(manifest_path)


@pytest.mark.parametrize(
    ("x1", "y1", "x2", "y2"),
    [
        (-0.1, 0.1, 0.5, 0.5),
        (0.1, -0.1, 0.5, 0.5),
        (0.1, 0.1, 1.1, 0.5),
        (0.1, 0.1, 0.5, 1.1),
        (0.5, 0.1, 0.5, 0.5),  # x1 == x2
        (0.6, 0.1, 0.5, 0.5),  # x1 > x2
        (0.1, 0.5, 0.5, 0.5),  # y1 == y2
        (0.1, 0.6, 0.5, 0.5),  # y1 > y2
        (float("nan"), 0.1, 0.5, 0.5),
        (0.1, float("inf"), 0.5, 0.5),
    ],
)
def test_rejects_invalid_normalized_boxes(x1: float, y1: float, x2: float, y2: float) -> None:
    with pytest.raises(ValidationError):
        EvaluationBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def test_rejects_wrong_media_sha256(tmp_path: Path) -> None:
    manifest_path = _create_minimal_dataset(tmp_path, write_media_files=True)
    p1 = tmp_path / "dataset" / "images" / "train_01.jpg"
    p1.write_bytes(b"corrupted-different-bytes")

    with pytest.raises(DatasetValidationError, match="SHA-256 mismatch"):
        load_evaluation_dataset(manifest_path)


def test_rejects_wrong_aggregate_sha256(tmp_path: Path) -> None:
    manifest_path = _create_minimal_dataset(
        tmp_path,
        custom_manifest_dict={"aggregateSha256": "e" * 64},
        write_media_files=True,
    )
    with pytest.raises(DatasetValidationError, match="aggregate SHA-256 mismatch"):
        load_evaluation_dataset(manifest_path)


@pytest.mark.parametrize(
    "traversal_split_path",
    [
        "../secret.jsonl",
        "/etc/passwd",
        r"C:\Windows\system32\cmd.exe",
    ],
)
def test_rejects_absolute_or_path_traversal_in_splits(
    tmp_path: Path, traversal_split_path: str
) -> None:
    manifest_path = _create_minimal_dataset(
        tmp_path,
        custom_manifest_dict={
            "splits": {
                "train": traversal_split_path,
                "validation": "indexes/validation.jsonl",
                "test": "indexes/test.jsonl",
            }
        },
        write_media_files=True,
    )
    with pytest.raises(DatasetValidationError, match="path traversal or absolute path"):
        load_evaluation_dataset(manifest_path)


@pytest.mark.parametrize(
    "traversal_media_path",
    [
        "../images/secret.jpg",
        "/var/data/image.jpg",
        r"D:\data\image.jpg",
    ],
)
def test_rejects_absolute_or_path_traversal_in_media_path(
    tmp_path: Path, traversal_media_path: str
) -> None:
    rows = [
        {
            "frameId": "f1",
            "mediaPath": traversal_media_path,
            "sha256": "0" * 64,
            "width": 640,
            "height": 480,
            "annotations": [],
        }
    ]
    manifest_path = _create_minimal_dataset(tmp_path, train_rows=rows, write_media_files=True)
    with pytest.raises(DatasetValidationError, match="path traversal or absolute path"):
        load_evaluation_dataset(manifest_path)


def test_rejects_missing_split_file(tmp_path: Path) -> None:
    manifest_path = _create_minimal_dataset(
        tmp_path,
        custom_manifest_dict={
            "splits": {
                "train": "indexes/nonexistent.jsonl",
                "validation": "indexes/validation.jsonl",
                "test": "indexes/test.jsonl",
            }
        },
        write_media_files=True,
    )
    with pytest.raises(DatasetValidationError, match="Split file does not exist"):
        load_evaluation_dataset(manifest_path)


def test_rejects_missing_media_file_when_verifying(tmp_path: Path) -> None:
    manifest_path = _create_minimal_dataset(
        tmp_path,
        write_media_files=False,  # media file won't be written on disk
    )
    with pytest.raises(DatasetValidationError, match="Media file does not exist"):
        load_evaluation_dataset(manifest_path)


def test_rejects_explicit_nulls() -> None:
    with pytest.raises(ValidationError):
        EvaluationFrame.model_validate(
            {
                "frameId": "f1",
                "mediaPath": None,
                "sha256": "0" * 64,
                "width": 640,
                "height": 480,
                "annotations": [],
            }
        )

    with pytest.raises(ValidationError):
        GroundTruthObject.model_validate(
            {
                "annotationId": None,
                "className": "Person",
                "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
            }
        )


def test_rejects_invalid_video_time_and_cross_field_invariants() -> None:
    # 1. frameIndex provided but videoTimeSeconds omitted
    with pytest.raises(ValidationError, match="Image frames must omit both"):
        EvaluationFrame.model_validate(
            {
                "frameId": "f1",
                "mediaPath": "img.jpg",
                "sha256": "0" * 64,
                "width": 640,
                "height": 480,
                "frameIndex": 1,
                "annotations": [],
            }
        )

    # 2. videoTimeSeconds provided but frameIndex omitted
    with pytest.raises(ValidationError, match="Image frames must omit both"):
        EvaluationFrame.model_validate(
            {
                "frameId": "f1",
                "mediaPath": "img.jpg",
                "sha256": "0" * 64,
                "width": 640,
                "height": 480,
                "videoTimeSeconds": 1.5,
                "annotations": [],
            }
        )

    # 3. negative videoTimeSeconds
    with pytest.raises(ValidationError):
        EvaluationFrame.model_validate(
            {
                "frameId": "f1",
                "mediaPath": "img.jpg",
                "sha256": "0" * 64,
                "width": 640,
                "height": 480,
                "frameIndex": 0,
                "videoTimeSeconds": -0.1,
                "annotations": [],
            }
        )

    # 4. negative frameIndex
    with pytest.raises(ValidationError):
        EvaluationFrame.model_validate(
            {
                "frameId": "f1",
                "mediaPath": "img.jpg",
                "sha256": "0" * 64,
                "width": 640,
                "height": 480,
                "frameIndex": -1,
                "videoTimeSeconds": 0.0,
                "annotations": [],
            }
        )


def test_rejects_exact_duplicate_hashes_crossing_splits(tmp_path: Path) -> None:
    shared_hash = "9" * 64
    train_rows = [
        {
            "frameId": "train_01",
            "mediaPath": "images/train.jpg",
            "sha256": shared_hash,
            "width": 640,
            "height": 480,
            "annotations": [],
        }
    ]
    test_rows = [
        {
            "frameId": "test_01",
            "mediaPath": "images/test.jpg",
            "sha256": shared_hash,
            "width": 640,
            "height": 480,
            "annotations": [],
        }
    ]
    manifest_path = _create_minimal_dataset(
        tmp_path, train_rows=train_rows, test_rows=test_rows, write_media_files=True
    )
    with pytest.raises(
        DatasetValidationError, match="Exact duplicate media SHA-256 found crossing splits"
    ):
        load_evaluation_dataset(manifest_path)


def test_ground_truth_ppe_episode_validation() -> None:
    # Valid episode
    ep = GroundTruthPpeEpisode(
        clip_id="clip_01",
        person_instance_id="person_1",
        ppe_item="HARD_HAT",
        start_time_seconds=1.0,
        end_time_seconds=3.5,
    )
    assert ep.clip_id == "clip_01"
    assert ep.end_time_seconds == 3.5

    # Valid with camelCase
    ep_camel = GroundTruthPpeEpisode.model_validate(
        {
            "clipId": "clip_02",
            "personInstanceId": 2,
            "ppeItem": "SAFETY_VEST",
            "startTimeSeconds": 0.0,
            "endTimeSeconds": 2.0,
        }
    )
    assert ep_camel.person_instance_id == 2

    # Negative startTime
    with pytest.raises(ValidationError):
        GroundTruthPpeEpisode(
            clip_id="clip_01",
            person_instance_id=1,
            ppe_item="HARD_HAT",
            start_time_seconds=-0.5,
            end_time_seconds=2.0,
        )

    # endTime < startTime
    with pytest.raises(ValidationError, match="endTimeSeconds must be greater than or equal to"):
        GroundTruthPpeEpisode(
            clip_id="clip_01",
            person_instance_id=1,
            ppe_item="HARD_HAT",
            start_time_seconds=2.0,
            end_time_seconds=1.0,
        )


def test_example_manifest_validates_against_model() -> None:
    example_path = (
        Path(__file__).resolve().parents[2] / "examples" / "ppe-evaluation-manifest.example.json"
    )
    if not example_path.exists():
        pytest.skip("Example manifest will be created during implementation")

    content = example_path.read_text(encoding="utf-8")
    manifest = EvaluationManifest.model_validate_json(content)
    assert manifest.dataset_id == "workersafety25"
    assert manifest.schema_version == "1.0.0"
    assert len(manifest.class_map) == 5


def test_manifest_rejects_non_canonical_or_duplicate_class_map() -> None:
    # 1. Non-canonical class name
    with pytest.raises(ValidationError, match="Canonical class map must contain exactly"):
        EvaluationManifest.model_validate(
            {
                "schemaVersion": "1.0.0",
                "datasetId": "ds1",
                "datasetVersion": "1",
                "sourceUrl": "https://example.com",
                "license": "MIT",
                "aggregateSha256": "0" * 64,
                "classMap": {"0": "Forklift"},
                "splits": {
                    "train": "indexes/train.jsonl",
                    "validation": "indexes/val.jsonl",
                    "test": "indexes/test.jsonl",
                },
            }
        )

    # 2. Duplicate canonical class name
    with pytest.raises(ValidationError, match="Canonical class map must contain exactly"):
        EvaluationManifest.model_validate(
            {
                "schemaVersion": "1.0.0",
                "datasetId": "ds1",
                "datasetVersion": "1",
                "sourceUrl": "https://example.com",
                "license": "MIT",
                "aggregateSha256": "0" * 64,
                "classMap": {
                    "0": "Person",
                    "1": "Person",
                    "2": "Hardhat",
                    "3": "NO-Hardhat",
                    "4": "Safety Vest",
                },
                "splits": {
                    "train": "indexes/train.jsonl",
                    "validation": "indexes/val.jsonl",
                    "test": "indexes/test.jsonl",
                },
            }
        )


def test_manifest_rejects_non_standard_splits() -> None:
    with pytest.raises(ValidationError, match="Splits must contain exactly"):
        EvaluationManifest.model_validate(
            {
                "schemaVersion": "1.0.0",
                "datasetId": "ds1",
                "datasetVersion": "1",
                "sourceUrl": "https://example.com",
                "license": "MIT",
                "aggregateSha256": "0" * 64,
                "classMap": {
                    "0": "Person",
                    "1": "Hardhat",
                    "2": "NO-Hardhat",
                    "3": "Safety Vest",
                    "4": "NO-Safety Vest",
                },
                "splits": {"foo": "indexes/x.jsonl"},
            }
        )


def test_manifest_and_loader_immutability(tmp_path: Path) -> None:
    manifest_path = _create_minimal_dataset(tmp_path)
    dataset = load_evaluation_dataset(manifest_path)

    # Manifest class_map must not be mutable
    with pytest.raises(TypeError):
        dataset.manifest.class_map["0"] = "Modified"  # type: ignore[index]

    # Manifest splits must not be mutable
    with pytest.raises(TypeError):
        dataset.manifest.splits["train"] = "modified.jsonl"  # type: ignore[index]

    # Dataset splits must not be mutable
    with pytest.raises(TypeError):
        dataset.splits["train"] = ()  # type: ignore[index]


def test_annotations_bounded_per_frame() -> None:
    box = {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}
    # 500 annotations -> valid
    valid_anns = [
        {"annotationId": f"ann_{i}", "className": "Person", "boundingBox": box} for i in range(500)
    ]
    frame = EvaluationFrame.model_validate(
        {
            "frameId": "f1",
            "mediaPath": "img.jpg",
            "sha256": "0" * 64,
            "width": 640,
            "height": 480,
            "annotations": valid_anns,
        }
    )
    assert len(frame.annotations) == 500

    # 501 annotations -> exceeds MAX_ANNOTATIONS_PER_FRAME
    invalid_anns = [
        {"annotationId": f"ann_{i}", "className": "Person", "boundingBox": box} for i in range(501)
    ]
    with pytest.raises(ValidationError, match="annotations"):
        EvaluationFrame.model_validate(
            {
                "frameId": "f1",
                "mediaPath": "img.jpg",
                "sha256": "0" * 64,
                "width": 640,
                "height": 480,
                "annotations": invalid_anns,
            }
        )


def test_person_instance_id_bounds() -> None:
    box = EvaluationBoundingBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2)
    # Valid string
    obj = GroundTruthObject(
        annotationId="a1", className="Person", boundingBox=box, personInstanceId="person_01"
    )
    assert obj.person_instance_id == "person_01"

    # Valid int
    obj_int = GroundTruthObject(
        annotationId="a1", className="Person", boundingBox=box, personInstanceId=42
    )
    assert obj_int.person_instance_id == 42

    # String exceeding 64 chars
    with pytest.raises(ValidationError):
        GroundTruthObject(
            annotationId="a1", className="Person", boundingBox=box, personInstanceId="x" * 65
        )

    # Empty string
    with pytest.raises(ValidationError):
        GroundTruthObject(
            annotationId="a1", className="Person", boundingBox=box, personInstanceId=""
        )

    # Negative int
    with pytest.raises(ValidationError):
        GroundTruthObject(
            annotationId="a1", className="Person", boundingBox=box, personInstanceId=-1
        )


def test_episode_bounded_and_restricted_ppe_items() -> None:
    # Invalid ppeItem
    with pytest.raises(ValidationError, match="ppeItem"):
        GroundTruthPpeEpisode(
            clipId="c1",
            personInstanceId=1,
            ppeItem="GLOVES",
            startTimeSeconds=0.0,
            endTimeSeconds=1.0,
        )

    # Valid items
    ep1 = GroundTruthPpeEpisode(
        clipId="c1",
        personInstanceId=1,
        ppeItem="HARD_HAT",
        startTimeSeconds=0.0,
        endTimeSeconds=1.0,
    )
    assert ep1.ppe_item == "HARD_HAT"

    ep2 = GroundTruthPpeEpisode(
        clipId="c1",
        personInstanceId=1,
        ppeItem="SAFETY_VEST",
        startTimeSeconds=0.0,
        endTimeSeconds=1.0,
    )
    assert ep2.ppe_item == "SAFETY_VEST"

    # frameIndex upper bound
    with pytest.raises(ValidationError):
        EvaluationFrame.model_validate(
            {
                "frameId": "f1",
                "mediaPath": "img.jpg",
                "sha256": "0" * 64,
                "width": 640,
                "height": 480,
                "frameIndex": 10_000_001,
                "videoTimeSeconds": 0.0,
                "annotations": [],
            }
        )


def test_aggregate_sha256_tamper_sensitive_and_order_independent() -> None:
    box1 = {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}
    f1 = EvaluationFrame.model_validate(
        {
            "frameId": "f1",
            "mediaPath": "img1.jpg",
            "sha256": "1" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "a1",
                    "className": "Person",
                    "boundingBox": box1,
                    "visibility": 1.0,
                }
            ],
        }
    )
    f2 = EvaluationFrame.model_validate(
        {
            "frameId": "f2",
            "mediaPath": "img2.jpg",
            "sha256": "2" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "a2",
                    "className": "Hardhat",
                    "boundingBox": box1,
                    "visibility": 1.0,
                }
            ],
        }
    )

    base_splits = {"train": (f1, f2)}
    base_hash = compute_dataset_aggregate_sha256(base_splits)

    # 1. Order-independent: swapped input frames produce identical hash
    swapped_splits = {"train": (f2, f1)}
    assert compute_dataset_aggregate_sha256(swapped_splits) == base_hash

    # 2. Tamper-sensitive: change visibility in f1
    f1_tampered = EvaluationFrame.model_validate(
        {
            "frameId": "f1",
            "mediaPath": "img1.jpg",
            "sha256": "1" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "a1",
                    "className": "Person",
                    "boundingBox": box1,
                    "visibility": 0.9,
                }
            ],
        }
    )
    tampered_splits = {"train": (f1_tampered, f2)}
    assert compute_dataset_aggregate_sha256(tampered_splits) != base_hash

    # 3. Tamper-sensitive: change bounding box
    box_altered = {"x1": 0.11, "y1": 0.1, "x2": 0.2, "y2": 0.2}
    f1_box_tampered = EvaluationFrame.model_validate(
        {
            "frameId": "f1",
            "mediaPath": "img1.jpg",
            "sha256": "1" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "a1",
                    "className": "Person",
                    "boundingBox": box_altered,
                    "visibility": 1.0,
                }
            ],
        }
    )
    assert compute_dataset_aggregate_sha256({"train": (f1_box_tampered, f2)}) != base_hash


def test_loader_always_requires_media_file_to_exist(tmp_path: Path) -> None:
    manifest_path = _create_minimal_dataset(tmp_path, write_media_files=False)
    with pytest.raises(DatasetValidationError, match="Media file does not exist"):
        load_evaluation_dataset(manifest_path)


def test_related_person_annotation_validation(tmp_path: Path) -> None:
    # 1. Negative: relatedPersonAnnotationId references non-existent ID
    rows_nonexistent = [
        {
            "frameId": "f1",
            "mediaPath": "images/train_01.jpg",
            "sha256": "a" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "ann_h1",
                    "className": "Hardhat",
                    "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
                    "relatedPersonAnnotationId": "ann_p_missing",
                }
            ],
        }
    ]
    p_nonexistent = _create_minimal_dataset(
        tmp_path / "d1", train_rows=rows_nonexistent, write_media_files=True
    )
    with pytest.raises(
        DatasetValidationError, match="references non-existent relatedPersonAnnotationId"
    ):
        load_evaluation_dataset(p_nonexistent)

    # 2. Negative: relatedPersonAnnotationId self-reference
    rows_self = [
        {
            "frameId": "f1",
            "mediaPath": "images/train_01.jpg",
            "sha256": "a" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "ann_p1",
                    "className": "Hardhat",
                    "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
                    "relatedPersonAnnotationId": "ann_p1",
                }
            ],
        }
    ]
    p_self = _create_minimal_dataset(tmp_path / "d2", train_rows=rows_self, write_media_files=True)
    with pytest.raises(DatasetValidationError, match="cannot reference itself"):
        load_evaluation_dataset(p_self)

    # 3. Negative: relatedPersonAnnotationId references non-Person class
    rows_non_person = [
        {
            "frameId": "f1",
            "mediaPath": "images/train_01.jpg",
            "sha256": "a" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "ann_v1",
                    "className": "Safety Vest",
                    "boundingBox": {"x1": 0.1, "y1": 0.3, "x2": 0.2, "y2": 0.4},
                },
                {
                    "annotationId": "ann_h1",
                    "className": "Hardhat",
                    "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
                    "relatedPersonAnnotationId": "ann_v1",
                },
            ],
        }
    ]
    p_non_person = _create_minimal_dataset(
        tmp_path / "d3", train_rows=rows_non_person, write_media_files=True
    )
    with pytest.raises(DatasetValidationError, match="references non-Person annotation"):
        load_evaluation_dataset(p_non_person)

    # 4. Negative: personInstanceId mismatch between related person and item
    rows_mismatch = [
        {
            "frameId": "f1",
            "mediaPath": "images/train_01.jpg",
            "sha256": "a" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "ann_p1",
                    "className": "Person",
                    "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9},
                    "personInstanceId": 1,
                },
                {
                    "annotationId": "ann_h1",
                    "className": "Hardhat",
                    "boundingBox": {"x1": 0.15, "y1": 0.12, "x2": 0.35, "y2": 0.25},
                    "personInstanceId": 2,  # Mismatched!
                    "relatedPersonAnnotationId": "ann_p1",
                },
            ],
        }
    ]
    p_mismatch = _create_minimal_dataset(
        tmp_path / "d4", train_rows=rows_mismatch, write_media_files=True
    )
    with pytest.raises(
        DatasetValidationError, match="does not match related person's personInstanceId"
    ):
        load_evaluation_dataset(p_mismatch)


def test_manifest_and_jsonl_line_size_limits(tmp_path: Path) -> None:
    # 1. Manifest exceeds size limit
    manifest_path = _create_minimal_dataset(tmp_path / "ds_big_m", write_media_files=True)
    # Append padding to exceed the bounded manifest size.
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(" " * (MAX_MANIFEST_BYTES + 10))

    with pytest.raises(DatasetValidationError, match="exceeds maximum allowed size"):
        load_evaluation_dataset(manifest_path)

    # 2. JSONL line exceeds its bounded size.
    ds_dir = tmp_path / "ds_big_line"
    p2 = _create_minimal_dataset(ds_dir, write_media_files=True)
    train_file = ds_dir / "dataset" / "indexes" / "train.jsonl"
    big_comment_line = " " * (MAX_JSONL_LINE_BYTES + 10) + "\n"
    train_file.write_text(big_comment_line, encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="exceeds maximum allowed size"):
        load_evaluation_dataset(p2)


def test_manifest_locks_schema_url_and_mapping_bounds() -> None:
    example_path = (
        Path(__file__).resolve().parents[2] / "examples" / "ppe-evaluation-manifest.example.json"
    )
    baseline = json.loads(example_path.read_text(encoding="utf-8"))

    invalid_schema = dict(baseline, schemaVersion="1.0.1")
    with pytest.raises(ValidationError, match="schemaVersion"):
        EvaluationManifest.model_validate(invalid_schema)

    invalid_url = dict(baseline, sourceUrl="https://user:secret@example.com/dataset")
    with pytest.raises(ValidationError, match="credentials"):
        EvaluationManifest.model_validate(invalid_url)

    long_class_key = json.loads(json.dumps(baseline))
    long_class_key["classMap"] = {
        "x" * 65: "Person",
        "1": "Hardhat",
        "2": "NO-Hardhat",
        "3": "Safety Vest",
        "4": "NO-Safety Vest",
    }
    with pytest.raises(ValidationError, match="classMap source names"):
        EvaluationManifest.model_validate(long_class_key)

    long_split_path = json.loads(json.dumps(baseline))
    long_split_path["splits"]["test"] = f"indexes/{'x' * 505}.jsonl"
    with pytest.raises(ValidationError, match="Split 'test' path"):
        EvaluationManifest.model_validate(long_split_path)


def test_frame_media_kind_invariants() -> None:
    base = {
        "frameId": "f1",
        "sha256": "0" * 64,
        "width": 640,
        "height": 480,
        "annotations": [],
    }
    with pytest.raises(ValidationError, match="Image mediaPath"):
        EvaluationFrame.model_validate(
            dict(base, mediaPath="frame.jpg", frameIndex=0, videoTimeSeconds=0.0)
        )
    with pytest.raises(ValidationError, match="Video mediaPath"):
        EvaluationFrame.model_validate(dict(base, mediaPath="clip.mp4"))
    with pytest.raises(ValidationError, match="unsupported image/video extension"):
        EvaluationFrame.model_validate(dict(base, mediaPath="frame.bin"))


def test_loader_rejects_drive_relative_and_duplicate_json_paths(tmp_path: Path) -> None:
    drive_relative = _create_minimal_dataset(
        tmp_path / "drive",
        custom_manifest_dict={
            "splits": {
                "train": "indexes/train.jsonl",
                "validation": "indexes/validation.jsonl",
                "test": "C:escape.jsonl",
            }
        },
    )
    with pytest.raises(DatasetValidationError, match="path traversal or absolute path"):
        load_evaluation_dataset(drive_relative)

    duplicate_key_manifest = _create_minimal_dataset(tmp_path / "duplicate-key")
    raw = duplicate_key_manifest.read_text(encoding="utf-8")
    raw = raw.replace(
        '"schemaVersion": "1.0.0",',
        '"schemaVersion": "1.0.0", "schemaVersion": "1.0.0",',
        1,
    )
    duplicate_key_manifest.write_text(raw, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Duplicate JSON object key"):
        load_evaluation_dataset(duplicate_key_manifest)


def test_loader_audits_video_identity_and_presentation_order(tmp_path: Path) -> None:
    duplicate_identity_rows = [
        {
            "frameId": "v0-a",
            "mediaPath": "videos/camera.mp4",
            "sha256": "1" * 64,
            "width": 640,
            "height": 480,
            "frameIndex": 0,
            "videoTimeSeconds": 0.0,
            "annotations": [],
        },
        {
            "frameId": "v0-b",
            "mediaPath": "videos/camera.mp4",
            "sha256": "1" * 64,
            "width": 640,
            "height": 480,
            "frameIndex": 0,
            "videoTimeSeconds": 0.2,
            "annotations": [],
        },
    ]
    duplicate_manifest = _create_minimal_dataset(
        tmp_path / "duplicate-video",
        train_rows=duplicate_identity_rows,
    )
    with pytest.raises(DatasetValidationError, match="Duplicate video frame identity"):
        load_evaluation_dataset(duplicate_manifest)

    decreasing_pts_rows = [
        dict(duplicate_identity_rows[0], frameId="v0", frameIndex=0, videoTimeSeconds=1.0),
        dict(duplicate_identity_rows[1], frameId="v1", frameIndex=1, videoTimeSeconds=0.5),
    ]
    decreasing_manifest = _create_minimal_dataset(
        tmp_path / "decreasing-pts",
        train_rows=decreasing_pts_rows,
    )
    with pytest.raises(DatasetValidationError, match="must increase with frameIndex"):
        load_evaluation_dataset(decreasing_manifest)


def test_video_person_observability_and_stable_return_order(tmp_path: Path) -> None:
    person_box = {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9}
    missing_observability = [
        {
            "frameId": "video-frame",
            "mediaPath": "videos/camera.mp4",
            "sha256": "2" * 64,
            "width": 640,
            "height": 480,
            "frameIndex": 0,
            "videoTimeSeconds": 0.0,
            "annotations": [
                {
                    "annotationId": "person-1",
                    "className": "Person",
                    "boundingBox": person_box,
                    "personInstanceId": 1,
                }
            ],
        }
    ]
    missing_manifest = _create_minimal_dataset(
        tmp_path / "missing-observability",
        train_rows=missing_observability,
    )
    with pytest.raises(DatasetValidationError, match="must declare observablePpeItems"):
        load_evaluation_dataset(missing_manifest)

    ordered_rows = [
        dict(
            missing_observability[0],
            frameId="z-frame",
            annotations=[
                {
                    "annotationId": "z-person",
                    "className": "Person",
                    "boundingBox": person_box,
                    "personInstanceId": 1,
                    "observablePpeItems": ["SAFETY_VEST", "HARD_HAT"],
                }
            ],
        ),
        dict(
            missing_observability[0],
            frameId="a-frame",
            frameIndex=1,
            videoTimeSeconds=0.2,
            annotations=[
                {
                    "annotationId": "a-person",
                    "className": "Person",
                    "boundingBox": person_box,
                    "personInstanceId": 1,
                    "observablePpeItems": [],
                }
            ],
        ),
    ]
    valid_manifest = _create_minimal_dataset(
        tmp_path / "valid-video",
        train_rows=ordered_rows,
    )
    loaded = load_evaluation_dataset(valid_manifest)
    assert [frame.frame_id for frame in loaded.get_split("train")] == ["z-frame", "a-frame"]
    assert loaded.get_split("train")[0].annotations[0].observable_ppe_items == (
        "HARD_HAT",
        "SAFETY_VEST",
    )


def test_video_annotations_require_identity_and_person_relationship(tmp_path: Path) -> None:
    box = {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9}
    person_without_identity = [
        {
            "frameId": "person-without-identity",
            "mediaPath": "videos/camera.mp4",
            "sha256": "3" * 64,
            "width": 640,
            "height": 480,
            "frameIndex": 0,
            "videoTimeSeconds": 0.0,
            "annotations": [
                {
                    "annotationId": "person-1",
                    "className": "Person",
                    "boundingBox": box,
                    "observablePpeItems": ["HARD_HAT"],
                }
            ],
        }
    ]
    missing_identity_manifest = _create_minimal_dataset(
        tmp_path / "missing-identity",
        train_rows=person_without_identity,
    )
    with pytest.raises(DatasetValidationError, match="must declare personInstanceId"):
        load_evaluation_dataset(missing_identity_manifest)

    ppe_without_relationship = [
        {
            "frameId": "ppe-without-relationship",
            "mediaPath": "videos/camera.mp4",
            "sha256": "4" * 64,
            "width": 640,
            "height": 480,
            "frameIndex": 0,
            "videoTimeSeconds": 0.0,
            "annotations": [
                {
                    "annotationId": "hardhat-1",
                    "className": "Hardhat",
                    "boundingBox": box,
                    "personInstanceId": 1,
                }
            ],
        }
    ]
    missing_relationship_manifest = _create_minimal_dataset(
        tmp_path / "missing-relationship",
        train_rows=ppe_without_relationship,
    )
    with pytest.raises(DatasetValidationError, match="must declare relatedPersonAnnotationId"):
        load_evaluation_dataset(missing_relationship_manifest)


def test_loader_bounds_frames_per_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("smartsite_ai.evaluation.dataset.MAX_FRAMES_PER_SPLIT", 1)
    rows = [
        {
            "frameId": f"frame-{index}",
            "mediaPath": f"images/frame-{index}.jpg",
            "sha256": str(index + 1) * 64,
            "width": 640,
            "height": 480,
            "annotations": [],
        }
        for index in range(2)
    ]
    manifest_path = _create_minimal_dataset(tmp_path, train_rows=rows)
    with pytest.raises(DatasetValidationError, match="exceeds maximum frame count"):
        load_evaluation_dataset(manifest_path)


def test_manifest_serialization_roundtrip() -> None:
    """Regression: MappingProxyType must not break model_dump or model_dump_json."""
    manifest = EvaluationManifest.model_validate(
        {
            "schemaVersion": "1.0.0",
            "datasetId": "workersafety25",
            "datasetVersion": "1",
            "sourceUrl": "https://example.com/dataset",
            "license": "CC BY 4.0",
            "aggregateSha256": "a" * 64,
            "classMap": {
                "0": "Person",
                "1": "Hardhat",
                "2": "NO-Hardhat",
                "3": "Safety Vest",
                "4": "NO-Safety Vest",
            },
            "splits": {
                "train": "indexes/train.jsonl",
                "validation": "indexes/validation.jsonl",
                "test": "indexes/test.jsonl",
            },
        }
    )

    # model_dump(mode="json") must not raise PydanticSerializationError
    json_dict = manifest.model_dump(mode="json", by_alias=True)
    assert isinstance(json_dict["classMap"], dict)
    assert isinstance(json_dict["splits"], dict)
    assert json_dict["classMap"]["0"] == "Person"
    assert json_dict["splits"]["train"] == "indexes/train.jsonl"

    # model_dump_json() must not raise PydanticSerializationError
    json_str = manifest.model_dump_json(by_alias=True)
    assert '"classMap"' in json_str
    assert '"splits"' in json_str

    # Round-trip: re-validate from dump must produce identical manifest
    roundtripped = EvaluationManifest.model_validate(json_dict)
    assert roundtripped.dataset_id == manifest.dataset_id
    assert dict(roundtripped.class_map) == dict(manifest.class_map)
    assert dict(roundtripped.splits) == dict(manifest.splits)

    # Immutability must still hold after serialization round-trip
    with pytest.raises(TypeError):
        roundtripped.class_map["0"] = "Modified"  # type: ignore[index]
    with pytest.raises(TypeError):
        roundtripped.splits["train"] = "modified.jsonl"  # type: ignore[index]

    assert roundtripped.model_copy(deep=True) == roundtripped
    assert hash(roundtripped) == hash(manifest)


def test_frame_and_nested_annotation_serialization_roundtrip_omits_nulls() -> None:
    image = EvaluationFrame.model_validate(
        {
            "frameId": "image-1",
            "mediaPath": "images/image-1.jpg",
            "sha256": "1" * 64,
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "annotationId": "person-image-1",
                    "className": "Person",
                    "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9},
                }
            ],
        }
    )
    image_json = image.model_dump_json(by_alias=True)
    assert ":null" not in image_json
    assert "frameIndex" not in image_json
    assert "videoTimeSeconds" not in image_json
    assert "relatedPersonAnnotationId" not in image_json
    assert EvaluationFrame.model_validate_json(image_json) == image

    video = EvaluationFrame.model_validate(
        {
            "frameId": "video-1",
            "mediaPath": "videos/camera.mp4",
            "sha256": "2" * 64,
            "width": 640,
            "height": 480,
            "frameIndex": 0,
            "videoTimeSeconds": 0.0,
            "annotations": [
                {
                    "annotationId": "person-video-1",
                    "className": "Person",
                    "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9},
                    "personInstanceId": "worker-1",
                    "observablePpeItems": ["SAFETY_VEST", "HARD_HAT"],
                }
            ],
        }
    )
    video_json = video.model_dump_json(by_alias=True)
    assert ":null" not in video_json
    assert EvaluationFrame.model_validate_json(video_json) == video


def test_person_relationship_and_padded_identifiers_are_rejected() -> None:
    box = {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9}
    with pytest.raises(ValidationError, match="Person annotations must not declare"):
        GroundTruthObject.model_validate(
            {
                "annotationId": "person-1",
                "className": "Person",
                "boundingBox": box,
                "relatedPersonAnnotationId": "person-2",
            }
        )

    with pytest.raises(ValidationError, match="non-blank and unpadded"):
        GroundTruthObject.model_validate(
            {
                "annotationId": "  ",
                "className": "Person",
                "boundingBox": box,
            }
        )

    with pytest.raises(ValidationError):
        GroundTruthObject.model_validate(
            {
                "annotationId": "person-1",
                "className": "Person",
                "boundingBox": box,
                "personInstanceId": " worker-1 ",
            }
        )
