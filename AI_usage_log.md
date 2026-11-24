# G07 AI Usage Log

| Date / phase         | Tool/model           | Purpose                                                                  | Prompt or instruction summary                                                                                                                   | Output used / changed / rejected                                                                                                                                | Checked by                                             |
|----------------------|----------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| 21 Sep 2026 / L1     | ChatGPT, GPT-5.6 Sol | Interpret L1 and assist with the Python forward model.                   | Explain band-edge versus Fermi-referenced counts; help implement `analysis.py`.                                                                 | Suggested structure used as a starting point, run locally and checked against C gates; incorrect plot emphasis corrected (see ledger).                          | Wang Kehan                                             |
| 23 Sep 2026 / L2     | ChatGPT, GPT-5.6 Sol | Assist L2 inverse analysis and reproducibility.                          | Repeat averaging, independent normalisation, Hall zero crossing, DoS fitting, bootstrap and E_F trajectory.                                     | Code structure/debugging and interpretation used; fixed 42-index assertion rejected and replaced by a data-derived length.                                      | Kong Xiaole                                            |
| 24 Sep 2026 / L3     | ChatGPT, GPT-5.6 Sol | Assist L3 spatial inversion analysis and wafer specification assessment. | Explain zero-mean reconstruction, Tikhonov regularisation, point-spread resolution, repeat-noise estimation and the wafer acceptance criterion. | Assisted with interpreting the reconstructed 100-pixel conductivity map, evaluating spatial resolution and uncertainty, and explaining the final pass decision. | Sun Minghong                                           |
| 23 Nov 2026 / report | ChatGPT, GPT-6       | Revise report transitions, equations and Vancouver references.           | Expand report to about 2,400 words, improve logic and put equations on separate lines.                                                          | Draft prose/layout used; numerical values and figures retained from G07 pipeline and checkpoints.                                                               | Group to review science, word count and final wording. |
| 23 Nov 2026 / QA     | ChatGPT, GPT-6       | Inspect repository and run reproducibility checks.                       | Inspect phase checkpoints and run `reproduce.py` in a clean extracted repository.                                                               | Clean run exited 0; schema, L1 mobility, L2 trajectory length, L3 zero-mean/noise/decision assertions passed.                                                   | Wang Wanqing                                           |

## Detailed Verification — L1

The suggested Python structure was used as a starting point and run locally in VS Code. The implementation was checked against the official L1 gate values:

- C(0 eV) = 1.000
- C(0.10 eV) = 2.599
- C(0.20 eV) = 5.104

The exact G07 operating point from `L1_params.json` was used rather than the nearest CSV row.

AI assisted with explanation and initial coding structure. Numerical execution, gate checking, interpretation and repository setup were performed and checked by the student/group.

## Detailed Verification — L2

All code was run locally on the untouched G07 data. The following checks were recorded:

- The Hall zero-crossing remained between measurement indices 19 and 20 after normalisation.
- The central fit selected the cone model with Eg/kBT = 0.
- The best hard-gap candidate was Eg/kBT = 0.75 but was below the 1 kBT resolution threshold.
- A 200-replicate bootstrap with seed 6028 selected a hard gap in 23/200 replicates.
- The final `reproduce.py` run regenerated the L1 and L2 outputs successfully.

AI assisted with code structure, debugging, interpretation and figure/checkpoint preparation. Numerical results were generated and verified by the committed local Python pipeline rather than entered manually.
