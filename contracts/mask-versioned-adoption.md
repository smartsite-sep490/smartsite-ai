# Mask camera protocol foundation (not operational)

`schemas/v1.8/technical-observation-event.json` adds `MASK` to explicit CAMERA
PPE observations. REGION observations retain the existing eight-family schema;
Harness and Hook remain PRESENT_ONLY. `schemas/v2.1/camera-policy-configuration-bundle.json`
permits six unique explicit required items, including Mask, at camera and region
scope. Inherit and disabled policy semantics are unchanged.

These are **opt-in canonical artifact definitions only**. Existing public
validators, ingestion, policy constants, database constraints, clients, UI
capabilities and AI runtime do not accept/select the new versions yet. Schema
validation alone does not prove same-person ownership, geometry, continuity,
bundle hashing, model capability or Backend admission. Do not register an
Ajv-only validator as a replacement for the semantic admission checks.

Released event schemas through1.7 and configuration schemas1.0/2.0 are unchanged.
Old readers must reject the new versions; changing `PPE_ITEMS` globally at this
stage would silently widen old policy decoders and is deliberately avoided.
No table, migration, role, runtime dependency, model or operational flag changes
belong to this contribution.

Synthetic examples pin these canonical SHA-256 values:

| Artifact                                | Hash                                                               |
| --------------------------------------- | ------------------------------------------------------------------ |
| v1.8 Mask event                         | `c3df004996d6f2d70003393e9bea7efd323e2baedb93a097fe9bd79d41527ae1` |
| v2.1 bundle body (without bundleSha256) | `05f703a6aa84ff9c181b66cd41e5aa9445c80db1d18b73a177614b10f42359a9` |

## Required staged adoption

1. Merge the reviewed canonical artifacts here; vendor identical bytes from the
   resulting immutable Main SHA into `smartsite-ai`, preserving provenance.
2. Add explicitly version-aware shared parsers/types and Backend old/new readers.
   Preserve descriptor/size, geometry, cross-Site, original-person, continuity,
   raw-payload and hash checks. Do not rewrite old history or version pins.
3. Review and test forward migrations for the existing policy, bundle, wire and
   notification constraints. Add Mask evaluator/notification handling while
   preserving subject episodes and grouped inbox revisions. Add no new tables.
   A rollout may not issue2.1/1.8 while any consumer/storage guard still rejects it.
4. AI adopts the new versions explicitly. Only a supported/qualified profile may
   advertise Mask; only explicit `NO-Mask` evidence can reach temporal Missing
   confirmation. Absence/ambiguity remains unknown. Legacy profiles stay unchanged.
5. Enable the Web picker after consumer/schema capability is available, then
   verify paired contracts, real PostgreSQL, grouping, rendering and failure paths.
6. Operational DB/model/camera rollout requires its separate review. Canonical
   artifact publication is not runtime activation, model qualification or a claim
   that all eight PPE families are complete. Harness/Hook still cannot be required
   as a Missing rule or treated as proof of correct fit or anchorage.
