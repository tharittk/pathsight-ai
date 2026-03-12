# Data plane

### Containter.
A 3D regular grid represented by `xarray` library.
The constructor takes (x_min, x_max), (y_min, y_max), (z_min, z_max), (dx, dy, dz).
`xarray` allows multiple values to be stored in (i, j, k). These values can be retrived
through indexing (i, j, k) or by label (x, y, z, method='nearest neighbor).
The value in each (i, j, k) shall be:
    - type: fault, reservoir, geobody
    - thickness (the value we are optimizing over)

