# Experimental positive PPE position filter

`PpePipeline(head_item_max_relative_center_y=...)` is an explicit code-level experiment, defaultNone. Existing callers/profile defaults are unchanged; there is no environment/runtime activation in this slice.

After unique-person association, positive Hat/Goggles observations below the configured relative head region, or on an unobservable cropped person, are omitted. Omission is UNKNOWN, never missing or permission. Valid positive evidence remains when a spare object is carried; contradictory raw positive/negative evidence stays unresolved. Carried objects cannot clear an already confirmed missing episode merely by being inside the person's whole-body box.

Synthetic tests use1/3 only to exercise boundaries. That value is NOT a calibrated safety threshold. An upright full-body box cannot establish head landmarks, physical wearing, fit or identity; crouching, partial visibility, tilted cameras and crowds require independent qualification. Other PPE association rules are unchanged. Explicit negative classes remain technical evidence and still need model/scene validation.

Default runtime continues its prior object-association semantics and therefore still has the known carried-item limitation. Do not enable this experiment or advertise wearing accuracy without frozen independent detection/association/retention evaluation. No model loading/training/website adoption accompanies publication.

## Current diagnostic disposition

HOLD: the scoped human-approved nine-relation coordinate control onC01/C02/C04 retained9/9 with legacy association and7/9 with this opt-in filter. C02Hat/Goggles were omitted because the Person box touches the image bottom, although the head/eye objects are visible. This is synthetic coordinate injection, not model inference or independent qualification. Do not tune against these three seen photos; qualify body-part observability and retention before activation. No label/split/consent scope changed.
