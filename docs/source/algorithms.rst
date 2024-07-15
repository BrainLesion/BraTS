Algorithms
========================

The following classes provide the top performing algorithms from several BraTS challenges.
All algorithms can be used for a single subject inference (one set of MRI scans) or for a batch of multiple subjects with the respective methods inherited from the base class (BraTSAlgorithm).


Challenge Algorithms
--------------------------------------


.. autoclass:: brats.AdultGliomaSegmenter


.. autoclass:: brats.MeningiomaSegmenter


.. autoclass:: brats.PediatricSegmenter


Abstract Base Class
--------------------------------------

.. autoclass:: brats.algorithms.BraTSAlgorithm
    



