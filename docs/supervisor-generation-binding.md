# Supervisor credential binding for one camera

The existing `smartsite-ai-camera-runtime --manifest <absolute reviewed manifest>` retains default behavior. A trusted supervisor can additionally pass allthree: `--generation-credential-file <fresh absolute artifact> --expected-camera-id <UUID> --expected-outbox <existing absolute outbox>`.

This override admits exactlyone camera and must match its configured camera and outbox. Only generationCredentialFile changes in memory through the strict manifest validator. The original manifest, source/model/policy/evidence paths and outbox bytes are not rewritten. Incomplete options, another camera/outbox, relative paths or a multi-camera template are refused before preflight/model/source opening.

The existing credential loader, one-time SQLite claim, expiry/session checks and configuration ACK remain authoritative. Passing a path proves neither issuance/commit nor readiness. The caller must await trusted Backend issuance first, retain outbox and avoid overlapping children. No DB/issuer credential belongs in the worker environment.

OwnerapprovedG4prototypeonly. No service installation, model/camera activation, runtime rollout or production claim. Cross-process supervisor ownership and real isolatedPGrenewal/old-generation fencing remain separate open gates. Tests use strict fixtures and mocked preflight, no model/nativecamera.

Supervisor-bound runs disable automatic dotenv discovery; worker settings come from its explicitly filtered environment. Legacy commands retain their prior settings behavior. This prevents a local `.env` from reintroducing issuer/service credentials or unrelated runtime switches after environment filtering.
