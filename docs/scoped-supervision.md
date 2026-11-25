# Experimental annotation scope admission

`verify_prepared_dataset(data_config, expected_profile="unified-ppe-15")`
adds an opt-in integrity check for a prepared experimental dataset. Omitting the
argument, or using `legacy-ppe-5`, preserves the existing five-class check.
The training CLI still accepts only its existing legacy dataset: this admission
API alone **does not enable partial-label training** or replace the provider loss.

The preparation manifest must declare the exact unified fifteen-class map,
`experimentalOnly: true`, and nonblank `license` and `attribution` strings.
These strings record provenance; they do not establish legal permission or
independent annotation quality. Every file, including `supervision.scopes.jsonl`,
must match the manifest byte inventory and aggregate digest. Cross-split image
overlap is rejected in experimental mode even if `reject_split_overlap` is omitted.

Each image uses a flat `train/images/name.ext`, `val/images/name.ext` or optional
`test/images/name.ext` path, with exactly one corresponding split/labels/name.txt.
Train and val must both contain images. Nested image folders and colliding stems
are rejected rather than silently left without supervision.

Each JSONL row contains exactly:

```json
{"path":"train/images/example.jpg","imageSha256":"<actual image SHA-256>","scope":[false,false,false,false,false,false,false,false,false,true,false,false,false,false,false],"positiveOnlyClassId":9,"sourceId":"approved-source-reference","labelOrigin":"human-positive-box-review"}
```

`scope` has fifteen strict booleans, with at least one true value. A true value
admits annotation supervision for that object class; it is not a PPE violation
decision. `positiveOnlyClassId=-1` declares complete annotations within the
selected classes, including reviewed background. A mode from 0 to 14 admits
only positive boxes of that one selected class; its scope must select exactly
that class and its label file must contain a positive. Unreviewed background
must later be masked out of the training loss. Every positive label must fall
inside its image scope. All images require exactly one row bound to their bytes;
missing, extra, repeated, stale or malformed rows are errors.

The future trainer must bind these scopes to the actual images loaded by the
provider, mask unknown/background gradients for positive-only images, disable
mixed-image augmentation and avoid reporting partial-label mAP as independent
quality. Until that implementation is reviewed, use this API for data admission
only. No model, camera, threshold, holdout or production capability changes here.
