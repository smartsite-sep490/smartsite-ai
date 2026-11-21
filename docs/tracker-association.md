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

An independent optional `new_track_confidence` admits new tracks only from
observations at or above that threshold. Existing owners first match high-confidence
observations, then may match weaker observations geometrically. Weak observations
cannot create a new track or take ownership when they overlap current high boxes
or the prior boxes of owners assigned/retired in the first pass. This conservative
rule can omit a real occluded low-confidence person; it must not be presented as
a complete count of people. Empty frames and ambiguous ownership still end segment
continuity. Both options default to `None` and need an explicit evaluated profile;
neither establishes a Worker identity or bridges missing observations.
