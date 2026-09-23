# AI Usage Log

## 2026-09-21 — L1

**Tool/model:** ChatGPT, GPT-5.6 Sol

**Purpose:**  
Help interpret the official L1 procedure and assist with implementation
and checking of the Python forward model.

**Prompt/task:**  
Asked the tool to read the supplied course materials, explain the
band-edge and Fermi-referenced carrier-count models, and help implement
analysis.py.

**How the output was used:**  
The suggested Python structure was used as a starting point and run
locally in VS Code.

**Verification:**  
The implementation was checked against the official L1 gate values:
C(0 eV) = 1.000,
C(0.10 eV) = 2.599,
C(0.20 eV) = 5.104.
The exact G07 operating point from L1_params.json was used rather than
the nearest CSV row.

**Extent of contribution:**  
AI assisted with explanation and initial coding structure. Numerical
execution, gate checking, interpretation and repository setup were
performed and checked by the student/group.

## L2 inverse analysis — 2026-09-23

**Tool:** ChatGPT (GPT-5.6 Sol)

**Purpose:** Assisted with implementation and checking of the L2 zero-dimensional inverse analysis, including repeat averaging, independent normalisation of R_H and R_xx, Hall zero-crossing checks, cone and hard-gap model generation, nearest-curve fitting, the 15% model-selection rule, bootstrap uncertainty analysis, Fermi-level trajectory plotting, and integration of L2 into reproduce.py.

**Verification:** All code was run locally on the untouched G07 data. The Hall zero-crossing remained between measurement indices 19 and 20 after normalisation. The central fit selected the cone model with Eg/kBT = 0. The best hard-gap candidate was Eg/kBT = 0.75 but was below the 1 kBT resolution threshold. A 200-replicate bootstrap with seed 6028 selected a hard gap in 23/200 replicates. The final reproduce.py run regenerated the L1 and L2 outputs successfully.

**Extent:** AI assisted with code structure, debugging, interpretation and figure/checkpoint preparation. Numerical results were generated and verified by the committed local Python pipeline rather than entered manually.