# Drawing comparison report

## At a glance

This comparison looked at 141 text field(s) on the source drawing and compared them to the target drawing. Results: 130 field(s) matched closely; 10 showed different text; 1 were missing on the target; 9 appeared only on the target; 10 were only a partial text match. The system raised 30 review item(s) (3 high-priority and 7 medium-priority)—see the review list for specifics.

## Summary counts

| Measure | Count | What it means |
| --- | --- | --- |
| Fields on source drawing | 141 | How many fields we tried to match |
| Matched closely | 130 | Text lines that lined up well |
| Text changed | 10 | Same slot, different wording or values |
| Missing on target | 1 | Present on source, not found on target |
| Only on target | 9 | Found on target, not on source |
| Unclear matches | 0 | System was unsure—needs a person |

## Text that changed between drawings

| Field (if known) | On source drawing | On target drawing | Confidence |
| --- | --- | --- | --- |
| — | #12-14X1-1/4 SDS | #12-14X1-1/4 SDS W/WASHER | 100% |
| — | 4' - 0" | 7' - 0" | 100% |
| — | 4' - 0" | 5' - 0" | 100% |
| — | 1' - 0" | 3' - 0" | 100% |
| — | 1' - 0" | 1' - 0 1/16" | 100% |
| — | 1' - 0" | 1' - 0 1/16" | 100% |
| — | W/ WASHER (FASTENER # 17A) | W/ WASHER (FASTENER | 100% |
| — | 06/08/2024 | 08/06/2024 | 100% |
| — | APPROVED - 09/11/2024 | APPROVED - 12/31/2024 | 100% |
| — | APPROVED - 09/11/2024 | 10/11/2024 | 100% |

## Missing on the target drawing

| What we saw on source | Field type | Confidence |
| --- | --- | --- |
| F | ocr_region | 100% |

## Extra text on the target drawing

| What we saw on target | Field type | Confidence |
| --- | --- | --- |
| GIRT EL. 1.1 | ocr_region | 100% |
| E | ocr_region | 100% |
| 0' - 6" | ocr_region | 100% |
| 0' - 6"1' - 0" | ocr_region | 100% |
| INSULATION | ocr_region | 100% |
| 0' - 11 15/16" | ocr_region | 100% |
| 0 | ocr_region | 100% |
| ISSUED FOR CONSTRUCTION | ocr_region | 100% |
| GAD-06 - 24072 Enclosure General Arrangement Drawing, Rev. B | ocr_region | 100% |

## Partial matches (similar but not identical)

| Field (if known) | Source | Target | Status | Confidence |
| --- | --- | --- | --- | --- |
| — | W/WASHER FASTENER #17A | W/ WASHER (FASTENER # 17A) | Partial match | 94% |
| — | 8" | 0' - 8" | Partial match | 78% |
| — | 6" | 0' - 6" | Partial match | 78% |
| — | 6" 6" | 0' - 6" | Partial match | 80% |
| — | 6" | 0' - 6" | Partial match | 78% |
| — | 6" 6" | 0' - 6" | Partial match | 80% |
| — | 6" 6" | 0' - 6" | Partial match | 80% |
| — | TO TOP AND BOTTOM OF EACH GIRT) | TOP AND BOTTOM OF EACH GIRT) - ASSUMING VERTICAL INSTALLATION OF 2' X 4' | Partial match | 78% |
| — | W/ WASHER (FASTENER | (FASTENER # 17A) | Partial match | 79% |
| — | 8/15/2024 2:59:25 PM | 11/26/2024 4:09:14 PM | Partial match | 79% |

## Unclear matches (please review)

_No uncertain matches._

## Review flags

| Topic | Severity | Guidance | Related field | Details |
| --- | --- | --- | --- | --- |
| Unclear or partial match | Medium | Medium — worth confirming when you have time. | auto-115-31e8a8c8 | [rule:partial_match_review] Partial textual match; confirm semantic equivalence; value_similarity=0.634; label_certainty=0.780; match_confidence=0.788; engine=classification=partial_match; normalized_… |
| Text changed | Low | Low — informational; often safe to ignore. | auto-125-4ef33ca7 | [rule:changed_value] Value differs between source and target under aligned labels; value_similarity=0.800; label_certainty=0.780; match_confidence=1.000; engine=classification=changed_value; normalize… |
| Extra information on target | Low | Low — informational; often safe to ignore. | auto-131-07a5633d | [rule:extra_information] Target field has no aligned source counterpart; field_type=ocr_region; critical_field=False; match_confidence=1.000 |
| Extra information on target | Low | Low — informational; often safe to ignore. | auto-132-f9278ccf | [rule:extra_information] Target field has no aligned source counterpart; field_type=ocr_region; critical_field=False; match_confidence=1.000 |
| Text changed | Low | Low — informational; often safe to ignore. | auto-139-7b104777 | [rule:changed_value] Value differs between source and target under aligned labels; value_similarity=0.842; label_certainty=0.780; match_confidence=1.000; engine=classification=changed_value; normalize… |
| Text changed | Medium | Medium — worth confirming when you have time. | auto-140-2cc0115e | [rule:changed_value] Value differs between source and target under aligned labels; value_similarity=0.621; label_certainty=0.780; match_confidence=1.000; engine=classification=changed_value; normalize… |
| Extra information on target | Low | Low — informational; often safe to ignore. | auto-146-c249be31 | [rule:extra_information] Target field has no aligned source counterpart; field_type=ocr_region; critical_field=False; match_confidence=1.000 |
| Unclear or partial match | High | High — review soon; may affect correctness. | auto-28-0773c1a9 | [rule:partial_match_review] Partial textual match; confirm semantic equivalence; value_similarity=0.444; label_certainty=0.780; match_confidence=0.776; engine=classification=partial_match; normalized_… |
| Text changed | Low | Low — informational; often safe to ignore. | auto-3-5a950c5b | [rule:changed_value] Value differs between source and target under aligned labels; value_similarity=0.800; label_certainty=0.780; match_confidence=1.000; engine=classification=changed_value; normalize… |
| Extra information on target | Low | Low — informational; often safe to ignore. | auto-36-7b7e83ff | [rule:extra_information] Target field has no aligned source counterpart; field_type=ocr_region; critical_field=False; match_confidence=1.000 |
| Unclear or partial match | Low | Low — informational; often safe to ignore. | auto-4-f8f911f9 | [rule:partial_match_review] Partial textual match; confirm semantic equivalence; value_similarity=0.917; label_certainty=0.780; match_confidence=0.938; engine=classification=partial_match; normalized_… |
| Missing information | Low | Low — informational; often safe to ignore. | auto-40-3a4c56d7 | [rule:missing_information] Source field absent on target drawing; field_type=ocr_region; critical_field=False; match_confidence=1.000 |
| Extra information on target | Low | Low — informational; often safe to ignore. | auto-40-6e5120f8 | [rule:extra_information] Target field has no aligned source counterpart; field_type=ocr_region; critical_field=False; match_confidence=1.000 |
| Text changed | Low | Low — informational; often safe to ignore. | auto-43-72b52ba4 | [rule:changed_value] Value differs between source and target under aligned labels; value_similarity=0.857; label_certainty=0.780; match_confidence=1.000; engine=classification=changed_value; normalize… |
| Text changed | Low | Low — informational; often safe to ignore. | auto-44-e9f09fab | [rule:changed_value] Value differs between source and target under aligned labels; value_similarity=0.857; label_certainty=0.780; match_confidence=1.000; engine=classification=changed_value; normalize… |
| Unclear or partial match | High | High — review soon; may affect correctness. | auto-48-3857b0cd | [rule:partial_match_review] Partial textual match; confirm semantic equivalence; value_similarity=0.444; label_certainty=0.780; match_confidence=0.776; engine=classification=partial_match; normalized_… |
| Unclear or partial match | Medium | Medium — worth confirming when you have time. | auto-49-b38ea2e1 | [rule:partial_match_review] Partial textual match; confirm semantic equivalence; value_similarity=0.500; label_certainty=0.780; match_confidence=0.795; engine=classification=partial_match; normalized_… |
| Unclear or partial match | High | High — review soon; may affect correctness. | auto-50-7ea049cb | [rule:partial_match_review] Partial textual match; confirm semantic equivalence; value_similarity=0.444; label_certainty=0.780; match_confidence=0.776; engine=classification=partial_match; normalized_… |
| Unclear or partial match | Medium | Medium — worth confirming when you have time. | auto-51-07bcc779 | [rule:partial_match_review] Partial textual match; confirm semantic equivalence; value_similarity=0.500; label_certainty=0.780; match_confidence=0.795; engine=classification=partial_match; normalized_… |
| Unclear or partial match | Medium | Medium — worth confirming when you have time. | auto-52-d83283b2 | [rule:partial_match_review] Partial textual match; confirm semantic equivalence; value_similarity=0.500; label_certainty=0.780; match_confidence=0.795; engine=classification=partial_match; normalized_… |
| Extra information on target | Low | Low — informational; often safe to ignore. | auto-53-21174d47 | [rule:extra_information] Target field has no aligned source counterpart; field_type=ocr_region; critical_field=False; match_confidence=1.000 |
| Extra information on target | Low | Low — informational; often safe to ignore. | auto-54-6d883001 | [rule:extra_information] Target field has no aligned source counterpart; field_type=ocr_region; critical_field=False; match_confidence=1.000 |
| Unclear or partial match | Medium | Medium — worth confirming when you have time. | auto-69-fd9b769b | [rule:partial_match_review] Partial textual match; confirm semantic equivalence; value_similarity=0.544; label_certainty=0.780; match_confidence=0.779; engine=classification=partial_match; normalized_… |
| Extra information on target | Low | Low — informational; often safe to ignore. | auto-72-ba47256d | [rule:extra_information] Target field has no aligned source counterpart; field_type=ocr_region; critical_field=False; match_confidence=1.000 |
| Text changed | Low | Low — informational; often safe to ignore. | auto-76-bc7d0e8d | [rule:changed_value] Value differs between source and target under aligned labels; value_similarity=0.857; label_certainty=0.780; match_confidence=1.000; engine=classification=changed_value; normalize… |
| Text changed | Low | Low — informational; often safe to ignore. | auto-77-459efb41 | [rule:changed_value] Value differs between source and target under aligned labels; value_similarity=0.737; label_certainty=0.780; match_confidence=1.000; engine=classification=changed_value; normalize… |
| Text changed | Low | Low — informational; often safe to ignore. | auto-78-3235866b | [rule:changed_value] Value differs between source and target under aligned labels; value_similarity=0.737; label_certainty=0.780; match_confidence=1.000; engine=classification=changed_value; normalize… |
| Extra information on target | Low | Low — informational; often safe to ignore. | auto-81-22986fa6 | [rule:extra_information] Target field has no aligned source counterpart; field_type=ocr_region; critical_field=False; match_confidence=1.000 |
| Text changed | Low | Low — informational; often safe to ignore. | auto-86-dc33316d | [rule:changed_value] Value differs between source and target under aligned labels; value_similarity=0.844; label_certainty=0.780; match_confidence=1.000; engine=classification=changed_value; normalize… |
| Unclear or partial match | Medium | Medium — worth confirming when you have time. | auto-90-76e00330 | [rule:partial_match_review] Partial textual match; confirm semantic equivalence; value_similarity=0.514; label_certainty=0.780; match_confidence=0.788; engine=classification=partial_match; normalized_… |

## Additional notes

```
{'mode': 'full', 'source_path': 'C:\\Users\\Jack.Fisher\\OneDrive - Kiewit Corporation\\Desktop\\Document OCR Text Review\\drawing-compare\\data\\test_inputs\\20037673-94.03.48.100-000154.01.VDR.01.01.pdf', 'target_path': 'C:\\Users\\Jack.Fisher\\OneDrive - Kiewit Corporation\\Desktop\\Document OCR Text Review\\drawing-compare\\data\\test_inputs\\20037673-94.03.48.100-000154.02.VDR.02.01.pdf', 'ocr_provider': 'windows_ocr', 'job_metadata': {}}
```

---

_Comparison ID: `a2349136-c949-4a2e-b3e0-edd026d2166c`_
