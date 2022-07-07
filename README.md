**Simple halo finder for use on cosmological simulation density fields**\
Richard H Mebane, UCSC Computational Astrophysics Group

**USAGE**

The halo finder works by looking for spherical regions of the cosmological density field with a mean density above some critical density (typically 200 times the mean density of the Universe). We start by filtering around the most dense cells in the box (corresponding to the centers of halos) and finding the radius at which the spherical mean density drops below the critical density. We then repeat this process at cells of smaller densities, ignoring any cells which are already in a halo or are below the critical density.

**USEFUL FUNCTIONS**

**get_snapshot(n_snapshot, dataDir, simDir, size, length)**

Read in the density field of a Cholla snapshot (functions provided by Bruno Villasenor)\
n_snapshot -> snapshot number\
dataDir -> location of all simulation data\
simDir -> location of specific simulation snapshots (snapshots are located at dataDir + simDir)\
size -> number of cells on the side of the box (default 256)\
length -> size of one side of the box in kpc (default 500)\
Returns (den, z), where den is the 3d density field as a numpy array and z is the current redshift of the snapshot

**get_halos(density, box_len, z, overdensity, segments)**

Filter a density snapshot to find all halos satisfying the above criteria\
density -> numpy array of the density field in Msun / kpc^3, as returned by get_snapshot\
box_len -> size of one side of the box in kpc\
z -> snapshot redshift\
overdensity -> local overdensity required for a collapsed halo (default 200)\
segments -> number of segments to subdivide each cell on a side to determine fraction contained in a filter region\
Returns an array of halo objects as defined below

**class halo**\
For a give halo h...\
h.POS -> tuple containing the position of central cell of the halo in the simulation box\
h.RADIUS -> radius of the halo in kpc\
h.MASS -> mass of the halo in Msun\
h.cells(N, box_len) -> returns a list of tuples corresponding to the cells belonging to the halo in a box of length box_len with N cells on a side

ISSUES

If the halo finder finds an isolated cell above the critical density, it will identify this single cell as a halo. This issue is most noticeable in lower resolution boxes. For example, see the below plot for a comparison on the halo mass function found with this halo finder as compared to that found with ROCKSTAR. This comparison was done on a z=0 snapshot of a 50 Mpc 256^3 simulation.

[mf_compare.pdf](https://github.com/RMebane/halo_finder/files/9067704/mf_compare.pdf)

All of the very low mass halos found below ROCKSTAR's lower mass limit belong to a single cell within the simulation box. This is not necessarily a problem, as these single cell halos do satisfy the halo criteria presented above. The best option may be to just remove them after the fact. Enforcing that the center of a halo must also be at a local maximum in the density field also removes a number of these halos.

Another issue is the efficiency of the filtering algorithm. The code currently just looks at each cell one by one around a filtered region to determine the fraction of that cell contained within the region. This is relatively fast since we are only filtering around a relatively small number of cells which are above the critical density and don't already belong to a halo. If the code is run on larger boxes with higher resolutions, however, it may be more efficient to perform a convolution at each filter radius on the entire box, even if we are only looking at a limited number of regions.

