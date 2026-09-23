# AI Error Ledger

## Error 1 — Hard-coded L2 trajectory length

**Tool/task:**  
ChatGPT — implementation of the L2 results and reproducibility checks.

**Incorrect AI output:**  
The AI initially suggested adding the assertion:

`assert n_index == 42`

when writing the L2 Fermi-level trajectory to results.json.

**How the error was detected:**  
The Practical Methods Guide states that the trajectory length must equal the number of unique measurement indices. The same frozen pipeline must later be applied to fresh blind-test data, so assuming that every dataset contains exactly 42 indices is not justified.

**Correction:**  
The hard-coded assertion was removed. The code retains only the data-dependent check:

`assert len(results["L2"]["EF_trajectory_eV"]) == n_index`

where `n_index` is calculated directly from the unique measurement indices in the supplied dataset.

**Verification after correction:**  
For the regular G07 dataset, the pipeline still produced a trajectory of length 42 and reproduced the full L1 and L2 outputs successfully. The implementation is now also compatible with a blind dataset containing a different number of measurement indices.

**Physical / analysis reason:**  
Measurement index represents the sequence of experimental measurements rather than a fixed physical grid. The inverse-analysis pipeline should therefore infer the required trajectory length from the input data instead of assuming a dataset-specific value.

**Error tag:** index/alignment / hard-coded assumption