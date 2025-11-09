Introduction
============

**SCILD: Spatial Cellular communication Inference with Ligand Diffusion and transport model**

SCILD is an interpretable framework for inferring spatial cellular communication (CCC)
from spatial transcriptomics data. It explicitly models ligand diffusion and receptor binding
as a *cargo transport system* that experiences potential losses during transmission.

Conceptual Overview
-------------------

- Models ligand diffusion and receptor binding as spatial transport.
- Uses sparse tensor optimization to infer communication strengths.
- Considers competitive and spatial constraints for robust inference.
- Enables in silico perturbations (e.g., ligand-receptor knockouts) for mechanistic insights.

Algorithmic Highlights
----------------------

SCILD formulates CCC inference as a **high-dimensional tensor optimization problem**, combining:

- Sparse tensor vectorization  
- Lagrangian duality for efficient optimization  
- Iterative alternating updates for ligand-receptor signal reconstruction  

.. image:: images/algorithm.png
   :align: center
   :alt: SCILD algorithm
   :width: 600px

Outputs
-------

- Spatial CCC strength tensor between ligand-receptor pairs.
- Visualization of in silico perturbations.
