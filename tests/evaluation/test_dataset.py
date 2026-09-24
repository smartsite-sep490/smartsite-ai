import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.dataset import (
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
        p3 = images_dir / "test_01.jpg"
        h3 = _write_media(p3, b"test-video-frame-01") if write_media_files else "c" * 64
        test_rows = [
            {
                "frameId": "test_001",
                "mediaPath": "images/test_01.jpg",
                "sha256": h3,
                "width": 1920,
                "height": 1080,
                "frameIndex": 0,
                "videoTimeSeconds": 0.0,
                "annotations": [
                    {
                        "annotationId": "ann_test1_1",
                        "className": "Safety Vest",
                        "boundingBox": {"x1": 0.2, "y1": 0.3, "x2": 0.4, "y2": 0.6},
                        "personInstanceId": 3,
                        "visibility": 0.9,
                    },
                    {
                        "annotationId": "ann_test1_2",
                        "className": "NO-Safety Vest",
                        "boundingBox": {"x1": 0.5, "y1": 0.3, "x2": 0.7, "y2": 0.6},
                        "personInstanceId": 4,
                        "visibility": 1.0,
                    },
                ],
            }
        ]

    (indexes_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n", encoding="utf-8"
    )
    (indexes_dir / "val.jsonl").write_text(
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
            "validation": "indexes/val.jsonl",
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
    dataset = load_evaluation_dataset(manifest_path, verify_checksums=True)

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


def test_rejects_unknown_classes(tmp_path: Path) -> None:
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
    manifest_path = _create_minimal_dataset(
        tmp_path, train_rows=invalid_rows, write_media_files=False
    )
    with pytest.raises(DatasetValidationError, match="Unknown class 'Forklift'"):
        load_evaluation_dataset(manifest_path, verify_checksums=False)


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
    manifest_path = _create_minimal_dataset(tmp_path, train_rows=dup_rows, write_media_files=False)
    with pytest.raises(DatasetValidationError, match="Duplicate frame ID 'frame_dup'"):
        load_evaluation_dataset(manifest_path, verify_checksums=False)


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
        tmp_path, train_rows=dup_ann_rows, write_media_files=False
    )
    with pytest.raises(DatasetValidationError, match="Duplicate annotation ID 'dup_ann_1'"):
        load_evaluation_dataset(manifest_path, verify_checksums=False)


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
        load_evaluation_dataset(manifest_path, verify_checksums=True)


def test_rejects_wrong_aggregate_sha256(tmp_path: Path) -> None:
    manifest_path = _create_minimal_dataset(
        tmp_path,
        custom_manifest_dict={"aggregateSha256": "e" * 64},
        write_media_files=True,
    )
    with pytest.raises(DatasetValidationError, match="aggregate SHA-256 mismatch"):
        load_evaluation_dataset(manifest_path, verify_checksums=True)


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
        custom_manifest_dict={"splits": {"train": traversal_split_path}},
        write_media_files=False,
    )
    with pytest.raises(DatasetValidationError, match="path traversal or absolute path"):
        load_evaluation_dataset(manifest_path, verify_checksums=False)


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
    manifest_path = _create_minimal_dataset(tmp_path, train_rows=rows, write_media_files=False)
    with pytest.raises(DatasetValidationError, match="path traversal or absolute path"):
        load_evaluation_dataset(manifest_path, verify_checksums=False)


def test_rejects_missing_split_file(tmp_path: Path) -> None:
    manifest_path = _create_minimal_dataset(
        tmp_path,
        custom_manifest_dict={"splits": {"train": "indexes/nonexistent.jsonl"}},
        write_media_files=False,
    )
    with pytest.raises(DatasetValidationError, match="Split file does not exist"):
        load_evaluation_dataset(manifest_path, verify_checksums=False)


def test_rejects_missing_media_file_when_verifying(tmp_path: Path) -> None:
    manifest_path = _create_minimal_dataset(
        tmp_path,
        write_media_files=False,  # media file won't be written on disk
    )
    with pytest.raises(DatasetValidationError, match="Media file does not exist"):
        load_evaluation_dataset(manifest_path, verify_checksums=True)


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
        tmp_path, train_rows=train_rows, test_rows=test_rows, write_media_files=False
    )
    with pytest.raises(
        DatasetValidationError, match="Exact duplicate media SHA-256 found crossing splits"
    ):
        load_evaluation_dataset(manifest_path, verify_checksums=False)


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
