# Experimental overlap association

`IoUPersonTracker(association_iou_margin=...)` is an explicit internal experiment
for distinguishable overlapping Person detections. The default `None` preserves
the original unique-edge association and conservative retirement behavior.
No worker manifest, CLI default or production profile enables this option.

With the option set, a connected IoU-candidate component continues only when
every prior track and current detection has a reciprocal unique strongest match,
each winning score exceeds its other eligible row/column candidates by strictly
more than the positive finite margin, and the component has a complete
one-to-one mapping. A tie, insufficient separation, merge, split or incomplete
mapping retires the ambiguous owners. Unrelated unique components can continue.
Detection confidence does not decide assignment, and no detections are removed.

This changes only geometric association. Segment lifecycle still ends continuity
on a missing processed Person, skipped source sequence, non-increasing capture
time, source/session, view, dimensions or detector metadata change. Integer
track IDs and segment UUIDs are technical claims, never Worker identities. A
geometrically strong match can still be a different physical person; in particular
an unobserved crossing between frames has no ground-truth guarantee.

Synthetic tests cover distinguishable overlaps, confidence reordering, ties,
merges/splits, a bystander, missing frames and source-sequence gaps. Local replay
may measure fragmentation but cannot prove association accuracy or justify
combining already separated Backend cases. Before any runtime activation, a
reviewed profile must bind the margin and temporal limits, and evaluation must
measure wrong-person transfers, crowd/crossing/occlusion behavior and notification
rate against independent permitted ground truth. No model-quality, identity,
camera-performance or production acceptance follows from this internal option.
