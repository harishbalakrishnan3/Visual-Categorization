Varying Abstraction Model (VAM)
===============================

Reference: [WoGB05]_

Module : :download:`VAM <../../../models/VAM/vam.py>`

The above module contains the necessary functions that aid in implementing the Varying Abstraction Model of
Categorization. The documentation for these functions is described below.

.. automodule:: VAM.vam
    :members:


Example
-------

A Sample Script : :download:`sample <../../../models/VAM/sample.py>`

The above script illustrates how one can use the vam module. As an example, the above script generates the best fit
pseudo-exemplar model, the corresponding model parameters and the minimum SSE. For the optimization,
scipy.optimize.minimize is used to minimize the SSE. Note that the following results are slightly better than the
values reported in [WoGB05]_. This could be due to the different optimization tools used in the implementation of the
model by the authors and in the example script here.

Script output
^^^^^^^^^^^^^
**Min SSE**: *0.017690536590039677*

**Best w**:  *[0.18963272, 0.19145339, 0.17382783, 0.44508607]*

**Best c**:  *11.285734631083642*

**Best b**:  *[0.07793568, 0.92206432]*

The corresponding pseudo-exemplar category representations are:

**Category A**: *[[1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]*

**Category B**: *[[0, 0, 0, 1], [0.33, 0.67, 0.33, 0.0]]*

.. rubric:: References
.. [WoGB05] Vanpaemel, Wolf; Storms, Gert; Ons, Bart: A Varying Abstraction Model for Categorization. In: B. Bara, L. Barsalou, & M. Bucciarelli (Eds.), Proceedings of the 27th annual conference of the Cognitive Science Society (pp. 2277–2282). Mahwah, NJ: Lawrence Erlbaum.


