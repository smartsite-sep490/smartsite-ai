# Shared Backend and edge camera worker

Approved direction: one operator machine runs camera inference; deployed and local
Web clients call the same deployed Backend. The Backend retains Site permissions,
episode grouping, notifications and evidence access. Redis is not required for
this topology; existing PostgreSQL queues and the SQLite edge outbox remain.

## Acceptance and boundaries

- A separately versioned Linux worker image includes the locked vision profile
  and existing runtime command. The foundation API Dockerfile stays unchanged.
- CPU is the initial verified profile; CUDA12.6 uses the existing pinned extra
  and requires separate NVIDIA hardware validation.
- Weights, recordings, credentials and generation files remain mounted/private.
  Starting the image without arguments only displays help, without inference.
- RTSP/Linux camera or approved video is the container source. Windows webcam
  capture remains host-owned until an explicit bridge is integrated.
- The same Backend must receive events and serve both Web clients. Localhost on
  another computer is not this operator machine.
- Publishing an image does not activate a camera, qualify a model or migrate a
  cloud database. Missing generation/configuration/model remains unavailable.
- Shared original images and live previews require a separate authenticated
  transport; changing only the ingestion URL is insufficient.

## Delivery sequence

1. Package existing worker; add operator Compose/configuration templates and
   documentation. Verify real container imports, CLI, fail-closed missing input,
   non-root/read-only execution, and no network-dependent model initialization.
2. Build and publish an immutable CPU tag/digest, retaining build/source proof.
   Run source checks and image smoke checks; review and merge the focused PR.
3. Reuse/complete authenticated evidence and preview transport to the deployed
   Backend; validate remote scope/configuration/generation before activating.
4. Exercise one real source and two browser clients against that Backend:
   event -> grouped case -> original image -> correct scoped notification.

## Progress

- [x] Inspect current Main, worker lifecycle, Docker and locked extras.
- [x] Verify Redis inventory: PostgreSQL queues + SQLite outbox already exist.
- [ ] Worker image build and container validation.
- [ ] Immutable registry publication and pull instructions.
- [ ] Shared evidence/preview and Backend cutover qualification.
- [ ] Two-client end-to-end camera demonstration.
