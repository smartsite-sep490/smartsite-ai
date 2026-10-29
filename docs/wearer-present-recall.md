# Wearer measurement report2.1

Input batch remains2.0.0. Newly generated offline reports use2.1.0 and add per-item present_recall: correctPRESENT / visiblePRESENT references. Missing/UNKNOWN predictions remain in that denominator; no visible-positive references yields null with NO_VISIBLE_PRESENT_REFERENCE. Existing missing recall, coverage and false-claim counters stay separate. Deserialized reports validate the new rate against underlying counts.

This is an offline report-format change, not the AI observation/event/API schema. Strict report2.0 readers need upgrading; archived reports are not rewritten or treated as2.1. No inference, label/split, model thresholds or qualification authority changes. A caller-declared visible reference is not automatically human-reviewed GT. Model accepted remainsfalse and qualification remainsNOT_ESTABLISHED.
