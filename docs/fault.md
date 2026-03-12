### Fault

The purpose of this module is to
    - load the petrel-formatted fault as a raw points (x, y, z)
    Note that these points are sparsely picked along the 2D cross-sections. They form a 3D plane.
    - pre-process these sparsely picked fault points to dense (x, y, z) with smoothing
    - grow the reservoir away from the fault plane both dipping side and anti-dipping side
    - the pre_process() method should return 3 array: the pre-processed fault plane, the reservoir dip side, the reservour anti-dip side
    - all these three items are in 3 dimensional array encoded they (x, y, z)
    - integrate(label='', object) should take any 3-dimensional array (fault, reservoir) and put it into RegularGrid3D container
    (I think such function should be in container.py)