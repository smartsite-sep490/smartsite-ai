# Explicit experimental PPE profiles

The manifest executable `smartsite-ai-camera-runtime --manifest <absolute path>` can opt into `experimentalModelProfile: native-ppe-10` or `unified-ppe-15`, together with `observationSchemaVersion: 1.1.0`. Both are explicit: omitting the profile keeps the existing exact five-class loader and two-item observation pipeline. An expanded profile with the older consumer version fails validation before model/source opening.

The profile validates the artifact class map exactly, including IDs and names. It does not select weights, change thresholds, download a model, qualify accuracy or authorize deployment. The runtime uses the existing artifact-verification/provider boundary and ordinary loader; this unit adds no warmup inference.

`unified-ppe-15` recognizes the declared detector taxonomy for Person, Hardhat/NO-Hardhat, Safety Vest/NO-Safety Vest, Gloves/NO-Gloves, Boots/NO-Boots, Goggles/NO-Goggles, Mask/NO-Mask, fullbody-Harness and Lanyard-Hook. Only the five existing canonical observation families can be emitted: hard hat, vest, gloves, boots and goggles. Mask, harness and hook remain detector objects, not wearer-state or business-violation claims.

PPE association requires one observable person owner. Ambiguous or contradictory detections remain unknown. An absent object never supplies a missing observation. Native10 has no explicit NO-Vest family; unified15 does. Existing Backend Site/Zone policy remains authoritative.

This extraction preserves current partitioned and generation-bound delivery flags, original credentials/context, event identity guards and captured subject lifecycle. Their existing defaults and operational gates do not change. The single-camera legacy CLI and diagnostic realtime preview retain their existing profile behavior; expanded operation uses the manifest path above.

Source tests use fake detections, manifests and artifact documents. Real-model quality, wearing qualification, camera latency, physical tracking, live notifications and production deployment require separate evidence. The held head-position heuristic, model warmup, preview/media, capture recovery, Face HTTP and supervisor binding from the original aggregate PR are separate source units.
