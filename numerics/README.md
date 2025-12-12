# Numerical Code

This directory contains the Python code used to generate all numerical
results and figures appearing in the project thesis. The code is written
to support the analysis in the thesis and is not intended as a general
library.

The numerical work focuses on neutron-star structure in general
relativity, using the Tolman–Oppenheimer–Volkoff equations together with
various equations of state.

## Directory Overview

```
numerics/
├── data/ # Precomputed numerical datasets
│
├── ideal_fermi_gas/ # Relativistic zero-temperature Fermi gas
│
├── incompressible_star/ # Analytic incompressible star solution
│
├── ideal_neutron_stars/ # Ideal neutron-star models and stability
│
└── npemu/ # npeμ matter, realistic EoS, and TOV solver
```


## Contents

- **ideal_fermi_gas/**  
  Illustrates basic properties of a relativistic Fermi gas and its
  equation of state.

- **incompressible_star/**  
  Analytic solution of the TOV equations for a uniform-density star,
  used as a consistency check and pedagogical example.

- **ideal_neutron_stars/**  
  Numerical solutions of the TOV equations for idealized neutron-star
  models, including radial oscillations and stability analysis.

- **npemu/**  
  The main numerical component of the project. Implements
  beta-equilibrated $npe\mu$ matter using a relativistic mean-field
  model, constructs composite equations of state, solves the TOV
  equations, and compares the results with observational constraints.

## Code Provenance

Parts of the numerical code are adapted from existing academic work:

- The $npe\mu$ relativistic mean-field implementation is based on code
  developed by **L. Pogliano** (Master’s thesis), available via the
  Norwegian National Research Archive:  
  [Pogliano, *Master’s thesis*](https://nva.sikt.no/registration/0198e9e2b085-6c0d9f80-868e-4ee7-81ed-33151eb22ebd)

- The radial stability solver and parts of the ideal neutron-star code
  are adapted from work by **H. Sletmoen** (Master’s thesis), available at:  
  [Sletmoen, *Master’s thesis*](https://nva.sikt.no/registration/0198e9e37e05-8cfbb1be-ecc5-4287-893b-68f433b1bb2c),  
  with an accompanying public repository:  
  [hersle/master-thesis (GitHub)](https://github.com/hersle/master-thesis/)

- Observational equation-of-state bands are taken from published data
  sets (e.g. Ng *et al.*) and processed locally for use in the plots.
