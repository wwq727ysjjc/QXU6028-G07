# AI Error Ledger

## Error 1 — Incorrect interpretation of the main L1 plot

**Tool/task:**  
ChatGPT — interpretation of the L1 workflow.

**Incorrect AI output:**  
The AI initially suggested that the main L1 task was to generate four
RH-EF and Rxx-EF curves for the two carrier-counting models.

**How the error was detected:**  
The official Practical Methods Guide specifies that the L1 sweep should
plot

C(EF) = n_band / n_Fermi

against Fermi level, with the C = 2 and C = 4 crossings and the exact
operating point identified.

**Correction:**  
The four-curve workflow was replaced by the official correction-factor
sweep.

**Verification after correction:**  
The corrected implementation gave:
C(0 eV) = 1.000
C(0.10 eV) = 2.599
C(0.20 eV) = 5.104

which agrees with the supplied physics gates.

**Physical reason:**  
L1 tests how changing the lower integration limit changes the extracted
carrier population. Therefore the direct quantity of interest is
n_band/n_Fermi.