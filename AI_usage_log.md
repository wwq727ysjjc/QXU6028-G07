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