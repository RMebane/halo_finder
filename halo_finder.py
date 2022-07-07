import numpy as np
import math as m
from scipy import interpolate
from scipy import integrate
from scipy import optimize
from scipy import ndimage
import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import time
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as mpl

# Simulation data functions written by Bruno Villasenor
def load_binary_data(filename, dtype=np.float32): 
    """ 
    We assume that the data was written 
    with write_binary_data() (little endian). 
    """ 
    f = open(filename, "rb") 
    data = f.read() 
    f.close() 
    _data = np.fromstring(data, dtype) 
    if sys.byteorder == 'big':
        _data = _data.byteswap()
    return _data 

def load_analysis_data( n_file, input_dir ):
    file_name = input_dir + f'{n_file}_analysis.h5'
    file = h5.File( file_name, 'r' ) 

    data_out = {}
    attrs = file.attrs
    for key in attrs:
        data_out[key] = file.attrs[key][0]

    data_out['phase_diagram'] = {}  
    phase_diagram = file['phase_diagram']
    for key in phase_diagram.attrs:
        data_out['phase_diagram'][key] = phase_diagram.attrs[key][0]
    data_out['phase_diagram']['data'] = phase_diagram['data'][...] 
    return data_out

def get_domain_block( proc_grid, box_size, grid_size ):
    np_x, np_y, np_z = proc_grid
    Lx, Ly, Lz = box_size
    nx_g, ny_g, nz_g = grid_size
    dx, dy, dz = Lx/np_x, Ly/np_y, Lz/np_z
    nx_l, ny_l, nz_l = nx_g//np_x, ny_g//np_z, nz_g//np_z,

    nprocs = np_x * np_y * np_z
    domain = {}
    domain['global'] = {}
    domain['global']['dx'] = dx
    domain['global']['dy'] = dy
    domain['global']['dz'] = dz
    for k in range(np_z):
        for j in range(np_y):
            for i in range(np_x):
                pId = i + j*np_x + k*np_x*np_y
                domain[pId] = { 'box':{}, 'grid':{} }
                xMin, xMax = i*dx, (i+1)*dx
                yMin, yMax = j*dy, (j+1)*dy
                zMin, zMax = k*dz, (k+1)*dz
                domain[pId]['box']['x'] = [xMin, xMax]
                domain[pId]['box']['y'] = [yMin, yMax]
                domain[pId]['box']['z'] = [zMin, zMax]
                domain[pId]['box']['dx'] = dx
                domain[pId]['box']['dy'] = dy
                domain[pId]['box']['dz'] = dz
                domain[pId]['box']['center_x'] = ( xMin + xMax )/2.
                domain[pId]['box']['center_y'] = ( yMin + yMax )/2.
                domain[pId]['box']['center_z'] = ( zMin + zMax )/2.
                gxMin, gxMax = i*nx_l, (i+1)*nx_l
                gyMin, gyMax = j*ny_l, (j+1)*ny_l
                gzMin, gzMax = k*ny_l, (k+1)*ny_l
                domain[pId]['grid']['x'] = [gxMin, gxMax]
                domain[pId]['grid']['y'] = [gyMin, gyMax]
                domain[pId]['grid']['z'] = [gzMin, gzMax]
    return domain

def select_procid( proc_id, subgrid, domain, ids, ax ):
    domain_l, domain_r = domain
    subgrid_l, subgrid_r = subgrid
    if domain_l <= subgrid_l and domain_r > subgrid_l:
        ids.append(proc_id)
    if domain_l >= subgrid_l and domain_r <= subgrid_r:
        ids.append(proc_id)
    if domain_l < subgrid_r and domain_r >= subgrid_r:
        ids.append(proc_id)

def select_ids_to_load( subgrid, domain, proc_grid ):
    subgrid_x, subgrid_y, subgrid_z = subgrid
    nprocs = proc_grid[0] * proc_grid[1] * proc_grid[2]
    ids_x, ids_y, ids_z = [], [], []
    for proc_id in range(nprocs):
        domain_local = domain[proc_id]
        domain_x = domain_local['grid']['x']
        domain_y = domain_local['grid']['y']
        domain_z = domain_local['grid']['z']
        select_procid( proc_id, subgrid_x, domain_x, ids_x, 'x' )
        select_procid( proc_id, subgrid_y, domain_y, ids_y, 'y' )
        select_procid( proc_id, subgrid_z, domain_z, ids_z, 'z' )
    set_x = set(ids_x)
    set_y = set(ids_y)
    set_z = set(ids_z)
    set_ids = (set_x.intersection(set_y)).intersection(set_z )
    return list(set_ids)

def load_snapshot_data_distributed( nSnap, inDir, data_type, fields, subgrid,  precision, proc_grid,  box_size, grid_size, show_progess=True ):
    
    
    # Get the doamin domain_decomposition
    domain = get_domain_block( proc_grid, box_size, grid_size )
    
    # Find the ids to load 
    ids_to_load = select_ids_to_load( subgrid, domain, proc_grid )

    #print(("Loading Snapshot: {0}".format(nSnap)))
    #Find the boundaries of the volume to load
    domains = { 'x':{'l':[], 'r':[]}, 'y':{'l':[], 'r':[]}, 'z':{'l':[], 'r':[]}, }
    for id in ids_to_load:
        for ax in list(domains.keys()):
            d_l, d_r = domain[id]['grid'][ax]
            domains[ax]['l'].append(d_l)
            domains[ax]['r'].append(d_r)
    boundaries = {}
    for ax in list(domains.keys()):
        boundaries[ax] = [ min(domains[ax]['l']),  max(domains[ax]['r']) ]

    # Get the size of the volume to load
    nx = int(boundaries['x'][1] - boundaries['x'][0])    
    ny = int(boundaries['y'][1] - boundaries['y'][0])    
    nz = int(boundaries['z'][1] - boundaries['z'][0])    

    dims_all = [ nx, ny, nz ]
    data_out = {}
    data_out[data_type] = {}
    for field in fields:
        data_particels = False
        if field in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']: data_particels = True 
        if not data_particels: data_all = np.zeros( dims_all, dtype=precision )
        else: data_all = []
        added_header = False
        n_to_load = len(ids_to_load)
        for i, nBox in enumerate(ids_to_load):
            name_base = 'h5'
            if data_type == 'particles': inFileName = '{0}_particles.{1}.{2}'.format(nSnap, name_base, nBox)
            if data_type == 'hydro': inFileName = '{0}.{1}.{2}'.format(nSnap, name_base, nBox)
        
            inFile = h5.File( inDir + inFileName, 'r')
            available_fields = inFile.keys()
            head = inFile.attrs
            if added_header == False:
                #print( ' Loading: ' + inDir + inFileName )
                #print( f' Available Fields:  {available_fields}')
                for h_key in list(head.keys()):
                    if h_key in ['dims', 'dims_local', 'offset', 'bounds', 'domain', 'dx', ]: continue
                    data_out[h_key] = head[h_key][0]
                    #if h_key == 'current_z': print((' current_z: {0}'.format( data_out[h_key]) ))
                added_header = True
        
            if show_progess:
                terminalString  = '\r Loading File: {0}/{1}   {2}'.format(i, n_to_load, field)
                sys.stdout. write(terminalString)
                sys.stdout.flush() 
        
            if not data_particels:
                procStart_x, procStart_y, procStart_z = head['offset']
                procEnd_x, procEnd_y, procEnd_z = head['offset'] + head['dims_local']
                # Substract the offsets
                procStart_x -= boundaries['x'][0]
                procEnd_x   -= boundaries['x'][0]
                procStart_y -= boundaries['y'][0]
                procEnd_y   -= boundaries['y'][0]
                procStart_z -= boundaries['z'][0]
                procEnd_z   -= boundaries['z'][0]
                procStart_x, procEnd_x = int(procStart_x), int(procEnd_x)
                procStart_y, procEnd_y = int(procStart_y), int(procEnd_y)
                procStart_z, procEnd_z = int(procStart_z), int(procEnd_z)
                data_local = inFile[field][...]
                data_all[ procStart_x:procEnd_x, procStart_y:procEnd_y, procStart_z:procEnd_z] = data_local
            
            else:
                data_local = inFile[field][...]
                data_all.append( data_local )
        
        if not data_particels:
            # Trim off the excess data on the boundaries:
            trim_x_l = subgrid[0][0] - boundaries['x'][0]
            trim_x_r = boundaries['x'][1] - subgrid[0][1]  
            trim_y_l = subgrid[1][0] - boundaries['y'][0]
            trim_y_r = boundaries['y'][1] - subgrid[1][1]  
            trim_z_l = subgrid[2][0] - boundaries['z'][0]
            trim_z_r = boundaries['z'][1] - subgrid[2][1]  
            trim_x_l, trim_x_r = int(trim_x_l), int(trim_x_r) 
            trim_y_l, trim_y_r = int(trim_y_l), int(trim_y_r) 
            trim_z_l, trim_z_r = int(trim_z_l), int(trim_z_r) 
            data_output = data_all[trim_x_l:nx-trim_x_r, trim_y_l:ny-trim_y_r, trim_z_l:nz-trim_z_r,  ]
            data_out[data_type][field] = data_output
        else:
            data_all = np.concatenate( data_all )
            data_out[data_type][field] = data_all
        if show_progess: print("")
    return data_out

def get_snapshot(n_snapshot, simDir="500mpc_snapshot/", size=256, length=500):
    dataDir = '/data/groups/comp-astro/rmebane/'
    inDir = dataDir + simDir

    # data_type = 'hydro'
    data_type = 'particles'

    fields = ['density', 'pos_x']

    precision = np.float32

    Lbox = length * 1.0e3    #kpc/h
    proc_grid = [ 2, 2, 2]
    box_size = [ Lbox, Lbox, Lbox ]
    grid_size = [ size, size, size ] #Size of the simulation grid
    subgrid = [ [0, size], [0, size], [0, size] ] #Size of the volume to load
    data = load_snapshot_data_distributed( n_snapshot, inDir, data_type, fields, subgrid,  precision, proc_grid,  box_size, grid_size, show_progess=False )
    if(USE_CUPY):
        return (cupy.array(data[data_type]['density']), data['current_z'])
    return (data[data_type]['density'], data['current_z'])

# wrap around if x is <0 or >=N
def wrap(x, N):
    if(x >= N):
        return x - N
    if(x < 0):
        return N + x
    return x

def get_cells(center, radius, N, box_len):
    #   returns all the cells and the fractions of each cell enclosed by radius
    #   radius and box_len should be in the same units
    #   N -> number of cells on one side of the box
    res = []
    fracs = []
    cell_size = box_len / N
    #   maximum distance in coordinates we need to check
    max_d = int(np.ceil(radius / cell_size))
    for i in range(center[0] - max_d, center[0] + max_d):
        for j in range(center[1] - max_d, center[1] + max_d):
            for k in range(center[2] - max_d, center[2] + max_d):
                i = wrap(i, N)
                j = wrap(j, N)
                k = wrap(k, N)
                '''
                if(i >= N):
                    i = i - N
                if(j >= N):
                    j = j - N
                if(k >= N):
                    k = k - N
                if(i < 0):
                    i = N + i
                if(j < 0):
                    j = N + j
                if(k < 0):
                    k = N + k'''
                res.append( (i, j, k) )
                fracs.append(enclosed_frac(center, (i,j,k), radius, N, box_len))
    return (np.array(res), np.array(fracs))

def get_distance(a, b, box_len, N):
    #   returns the distance between a and b
    #   result will be in the same units as box_len
    #   N -> number of cells on one side of the box
    cell_size = box_len / N
    x = cell_size * abs(a[0] - b[0])
    y = cell_size * abs(a[1] - b[1]) 
    z = cell_size * abs(a[2] - b[2])
    #   make sure we are correctly wrapping around
    if(x > box_len / 2.0):
        x = box_len - x
    if(y > box_len / 2.0):
        y = box_len - y
    if(z > box_len / 2.0):
        z = box_len - z
    return np.sqrt(x**2.0 + y**2.0 + z**2.0)

def remove_halos(halos, N, box_len):
    #   remove any halos with centers inside another halo
    #   this is probably not necessary anymore
    rs = []
    ms = []
    res = []
    rejected = []

    for h in halos:
        rs.append(h.RADIUS)
        ms.append(h.MASS)
    rs = np.array(rs)
    ms = np.array(ms)

    # sort by radius first, then mass
    s = np.lexsort((ms, rs))
    for h in halos[s]:
        valid = True
        n_matches = 0
        n_rejected_matches = 0
        for g in halos:
            if(not coord_eq(h.POS, g.POS)):
                if(get_distance(h.POS, g.POS, box_len, N) < g.RADIUS):
                    n_matches += 1
                    valid = False
        for g in rejected:
            if(not coord_eq(h.POS, g.POS)):
                if(get_distance(h.POS, g.POS, box_len, N) < g.RADIUS):
                    n_rejected_matches += 1
        if(valid or n_matches == n_rejected_matches):
            res.append(h)
        else:
            rejected.append(h)
    return np.array(res)



def get_mass(center, radius, density, z, box_len):
    N = len(density)
    a = 1.0 / (1.0 + z)
    cell_volume = (1.0e3 * box_len / N) ** 3.0
    mass = 0
    t = get_cells(center, radius, N, box_len)
    for c,f in zip(t[0], t[1]):
        i = c[0]
        j = c[1]
        k = c[2]
        mass += density[i][j][k] * cell_volume * f 
    return mass

def rho_mean(cells, fracs, density):
    total = 0
    for c,f in zip(cells, fracs):
        total += density[c[0], c[1], c[2]] 
    return total / len(cells)

class halo:
  
    #   return an array of cells which belong to the given halo
    #   they may not be ordered in any way, haven't decided yet
    #   returns a tuple of arrays. first is the cells, second is the fraction of each cell contained in radius R
    def cells(self, N, box_len):
        return get_cells(self.POS, self.RADIUS, N, box_len)

    def __init__(self, MASS = 0.0, POS = (0, 0, 0), RADIUS = 0.0):
        self.MASS = MASS
        self.POS = POS
        self.RADIUS = RADIUS

def is_member(x, halos, box_len, N):
    #   checks if the given position is a member of a halo in halos
    for h in halos:
        if(get_distance(x, h.POS, box_len, N) < h.RADIUS):
            return True
    return False

def coord_eq(a, b):
    if(a[0] != b[0] or a[1] != b[1] or a[2] != b[2]):
        return False
    return True

def is_peak(x, density):
    #   Checks if x is a local peak in the density field
    N = len(density)
    for i in range(x[0]-1, x[0]+2):
        for j in range(x[1]-1, x[1]+2):
            for k in range(x[2]-1, x[2]+2):
                if(not coord_eq(x, (i, j, k))):
                    i = wrap(i, N)
                    j = wrap(j, N)
                    k = wrap(k, N)
                    '''
                    if(i >= N):
                        i = i - N
                    if(j >= N):
                        j = j - N
                    if(k >= N):
                        k = k - N
                    if(i < 0):
                        i = N + i
                    if(j < 0):
                        j = N + j
                    if(k < 0):
                        k = N + k'''
                    if(density[x[0], x[1], x[2]] < density[i, j, k]):
                        return False
    return True

def is_valid_center(x, halos, box_len, density, overdensity=200.):
    #   Returns true IFF x is not inside another halo and it is a local peak in the density field
    i = x[0]
    j = x[1]
    k = x[2]
    mean_density = np.mean(density)
    if(not(not is_member(x, halos, box_len, len(density) and density[i][j][k] > overdensity * mean_density))):
        return False
    return is_peak(x, density)

def get_halos(density, box_len, z, overdensity=200., segments=5):
    #   returns an array of halos from filtering a given density field
    #   INPUTS
    #   density -> 3d array of density field in units of Msun / kpc^3
    #   box_len -> size of the box in kpc
    #   z -> shapshot redshift
    #   overdensity -> overdensity of a halo relative to the mean
    #   segments -> number of subdivisions on one side of a cell to compute partial (probably a better way to do this, but this is at least fast)
    mean_density = np.mean(density)
    density_f = density.flatten()
    s = np.argsort(density_f)
    #   Sort cells by density, and start filtering around the densest regions
    #   Require that the center of a halo at least be above overdensity
    N_valid = len(density_f[density_f > overdensity * mean_density])
    indices = np.array(list(np.ndindex(density.shape)))
    ind_sort = np.flip(indices[s], axis=0)
    ind_sort = ind_sort[:N_valid]
    halos = []
    box_len = box_len / 1.0e3
    R_step = box_len / len(density) / segments
    for x in ind_sort:
        #   check if this point already belongs to a halo
        if(is_valid_center(x, halos, box_len, density, overdensity)):
            i = 1
            #   Start on small scales and move outward
            #   Maybe we can do this the other way around (i.e., large to small)?
            while i<len(density):
                R = i * R_step
                #   c -> array of cells contained within R
                #   f -> array of fractions of each cell contained within R
                c,f = get_cells(x, R, len(density), box_len)
                d = rho_mean(c, f, density)
                if(d < overdensity * mean_density):
                    h = halo(MASS = get_mass(x, R, density, z, box_len), POS = x, RADIUS = R)
                    halos.append(h)
                    break
                i += 1
    #   remove any halos inside a larger halo
    #   I don't think this should happen much anymore
    return remove_halos(np.array(halos), len(density), box_len)

