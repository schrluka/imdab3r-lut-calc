#!/usr/bin/env python
""" Create C code header from CSV

Parses a lookup table stored as CSV file and produces a C code header file to be compiled into firmware and/or
simulation code.
"""


import sys
import numpy as np
import csv_io
import time


__author__ = "Lukas Schrittwieser"
__copyright__ = "Copyright 2018, ETH Zurich"
__credits__ = ["Michael Leibl"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Lukas Schrittwieser"
__email__ = "schrittwieser@lem.ee.ethz.ch"
__status__ = "Production"


def export_c_header(fn, orig_fn, n_vars, dim, grid_res, ranges, data, prefix='lut_'):
    # make sure all dimension have the same number of entries which simplifies the C code (so only that is supported)
    for i in range(1,dim):
        if grid_res[0] != grid_res[i]:
            raise Exception('Number of sampling points not equal over dimensions!')
    #
    f = open(fn, mode='w')
    f.write('// IMDAB3R precalculated relative switching times\n')
    f.write('// Generated: {} from {}\n'.format(time.strftime("%d/%m/%Y - %H:%M:%S"),orig_fn))
    f.write('\n#include <stdint.h>\n')
    f.write('\n#ifndef __MOD_TABLE__\n')
    f.write('#define __MOD_TABLE__\n\n')

    f.write('#define {}n_vars {}\n'.format(prefix, n_vars))
    f.write('#define {}n_dim  {}\n\n'.format(prefix, dim))
    # write sampling grid positions!
    f.write('const uint32_t {}resolution[] = {{'.format(prefix))
    for r in ranges:
        f.write('{}, '.format(len(r)))
    f.write('};\n')
    #
    f.write('const float {}ranges[{}][{}] = {{\n'.format(prefix, dim, grid_res[0]))
    for i in range(0,dim):
        f.write('    {')
        for r in ranges[i]:
            f.write('{:.3e}F, '.format(r))
        f.write('},\n')
    f.write('};\n\n')
    #
    f.write('const float {}data[]'.format(prefix))
    f.write(' = { \n    ')
    ptr = np.zeros(dim, dtype=np.int)  # pointer (index) within data
    while True:
        # write one entry with n_vars values
        d = data[tuple(ptr)]
        #print(d)
        for i in range(0,n_vars):
            f.write('{:.5e}F, '.format(d[i]))
        f.write('\n    ')
        # increment pointer with correct wrap around
        csv_io.increment_wr_ptr(ptr,grid_res)
        if np.max(ptr) == 0:
            break   # all dimensions wrapped -> we are done
    f.write('};\n\n')
    f.write('#endif\n')
    f.close()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('give file name of input file as parameter (and only that)')
        exit(0)

    fn = sys.argv[1]
    res = csv_io.import_csv(fn)

    n_vars = res.n_vars
    data = res.data

    if n_vars == 5:
        # strip the useless g2l entry, hardware does not support it, this was just a legacy idea of Leibl
        print('stripping g2l entry from data')
        n_vars = 4
        data = np.delete(data, 2, 3)
        print('new size of data array: ', data.shape)

    export_c_header('lut.h', fn, n_vars, res.dim, res.grid_res, res.ranges, data)




