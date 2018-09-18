"""Read and write special CSV files

This file provides functions to import or export 3D lookup tables using a custom CSV file format.
"""
import numpy as np
from collections import namedtuple


__author__ = "Lukas Schrittwieser"
__copyright__ = "Copyright 2018, ETH Zurich"
__credits__ = ["Michael Leibl"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Lukas Schrittwieser"
__email__ = "schrittwieser@lem.ee.ethz.ch"
__status__ = "Production"


def increment_wr_ptr(ptr, grid_res):
    # increment write pointer (THE SEQUENCE OF THIS MUST MATCH export_csv!)
    for i in range(0, len(ptr)):  # i loops over all dimensions
        ptr[i] = ptr[i] + 1
        if ptr[i] >= grid_res[i]:
            # this dimension overflows
            ptr[i] = 0
        else:
            return  # no overflow, we are done


def export_csv(fn, grid_res, i_dc_range, u_dc_range, u_2_range, s):
    # f = open(fn,mode='w',newline='')
    f = open(fn, mode='w')
    # first line is the number of dimension and second is the number of output variables
    f.write('3,{}\n'.format(len(s[0,0,0,0:])))
    # next is the sampling grid, each dimension on an individual line
    for k in i_dc_range:
        f.write('{:.4e},'.format(k))
    f.write('\n')
    for k in u_dc_range:
        f.write('{:.4e},'.format(k))
    f.write('\n')
    for k in u_2_range:
        f.write('{:.4e},'.format(k))
    f.write('\n')
    # last part is the lookup table data itself
    for k3 in range(0, grid_res[2]):
        for k2 in range(0, grid_res[1]):
            for k1 in range(0, grid_res[0]):
                for v in s[k1, k2, k3, 0:]:
                    f.write('{:.5e}, '.format(v))
                f.write('\n')
    f.close()


def import_csv(fn):
    Result = namedtuple('Result', ['dim', 'n_vars', 'grid_res', 'ranges', 'data'])
    f = open(fn, mode='r')
    line = f.readline()
    if not line:
        raise Exception('First line missing')
    v = line.split(',', maxsplit=2)
    dim = int(v[0])
    n_vars = int(v[1])
    #if dim != 3:
    #    raise ValueError('Dimensionality error')
    #if n_vars != 5:
    #    raise ValueError('Number of variables wrong')
    grid_res = []
    ranges = []
    # read sampling grid values for each dimension
    for i in range(dim):
        line = f.readline()
        if not line:
            raise Exception('range line missing')
        new_range = []
        # parse all values in this line
        for v in line.split(','):
            v = v.strip()
            if not v:
                break
            new_range.append(float(v))
        ranges.append(new_range)
        grid_res.append(len(new_range))
    grid_res.append(n_vars) # make n_vars last entry of grid res to allow single value indexing of data
    # read the data
    # print(grid_res)
    data = np.zeros(grid_res)
    pt = []
    wr_ptr = np.zeros(dim, dtype=np.int)  # write pointer within data
    while True:
        line = f.readline()
        if not line:
            break
        # parse all values in this line
        for str in line.split(','):
            str = str.strip()
            if not str:
                break
            v = float(str)
            pt.append(v)    # each entry has n_vars float values
            if len(pt) == n_vars:
                # got one full point of data
                data[tuple(wr_ptr)] = pt    # copy it to our data array
                pt = [] # clear entries for next round
                # increment write pointer (THE SEQUENCE OF THIS MUST MATCH export_csv!)
                increment_wr_ptr(wr_ptr, grid_res)
                # for i in range(0,dim):  # i loops over all dimensions
                #     wr_ptr[i] = wr_ptr[i] + 1
                #     if wr_ptr[i] >= grid_res[i]:
                #         # this dimension overflows
                #         wr_ptr[i] = 0
                #     else:
                #         break   # no overflow, we are done
                # # print(wr_ptr)
    if len(pt) != 0:
        raise ValueError('data values left over in file')
    #print('wr_ptr: ',wr_ptr)
    if np.max(wr_ptr) > 0:
        print(wr_ptr)
        raise ValueError('number of data entries in file did not match the grid')
    # assemble everything as named tuple and return it
    res = Result(dim, n_vars, grid_res, ranges, data)
    return res

