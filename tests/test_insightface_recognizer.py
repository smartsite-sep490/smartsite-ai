from pathlib import Path

from cryptography.fernet import Fernet

from smartsite_ai.config import Settings
from smartsite_ai.inference.identity import UnavailableFaceRecognizer
from smartsite_ai.inference.insightface_recognizer import (
    EncryptedTemplateStore,
    InsightFaceDemoRecognizer,
    build_face_recognizer,
)


def test_template_store_encrypts_embeddings_and_round_trips(tmp_path: Path) -> None:
    store_path = tmp_path / "private" / "templates.fernet"
    store = EncryptedTemplateStore(store_path, Fernet.generate_key().decode("ascii"))

    store.upsert("fp_demo", [0.6, 0.8])

    assert store.all() == {"fp_demo": [0.6, 0.8]}
    assert b"fp_demo" not in store_path.read_bytes()


def test_demo_recognizer_requires_explicit_private_configuration(tmp_path: Path) -> None:
    disabled = build_face_recognizer(Settings(_env_file=None))
    missing_key = build_face_recognizer(
        Settings(identity_demo_mode=True, identity_model_root=tmp_path, _env_file=None)
    )

    assert isinstance(disabled, UnavailableFaceRecognizer)
    assert isinstance(missing_key, UnavailableFaceRecognizer)


def test_demo_recognizer_is_configured_without_loading_or_downloading_model(tmp_path: Path) -> None:
    recognizer = build_face_recognizer(
        Settings(
            identity_demo_mode=True,
            identity_model_root=tmp_path,
            identity_template_store_path=tmp_path / "private" / "templates.fernet",
            identity_template_encryption_key=Fernet.generate_key().decode("ascii"),
            _env_file=None,
        )
    )

    assert isinstance(recognizer, InsightFaceDemoRecognizer)
