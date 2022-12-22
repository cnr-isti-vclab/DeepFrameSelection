#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import numpy as np

#
#
#
def get3DPoints(name_file):
    with open(name_file, "r") as file:
         c = 0
         for x in file:
             c += 1
    c -= 3
    
    points = np.zeros((c, 3), dtype = float)
    
    with open(name_file, "r") as file:
        c = 0
        for x in file:
            if c > 2:
                lst = x.split(" ", 7)
         
                points[c - 3, 0] = lst[1]
                points[c - 3, 1] = lst[2]
                points[c - 3, 2] = lst[3]
            c += 1
         
    file.close()
    
    return points

#
#
#
def writeHeaderPLY(file_out, nv):
    file_out.write("ply\n")
    file_out.write("format ascii 1.0\n")
    file_out.write("element vertex " + str(nv) + "\n")
    file_out.write("property float x\n")
    file_out.write("property float y\n")
    file_out.write("property float z\n")
    file_out.write("property uchar red\n")
    file_out.write("property uchar green\n")
    file_out.write("property uchar blue\n")
    file_out.write("property float quality\n")
    file_out.write("element face 0\n")
    file_out.write("property list uchar int vertex_indices\n")
    file_out.write("end_header\n")

#
#
#
def writePLY(name_file):
    name, ext = os.path.splitext(name_file)
    name_out = name + ".ply"
    file_out = open(name_out, "w")

    with open(name_file, "r") as file:
         c = 0
         for x in file:
             c += 1
    c -= 3
    
    writeHeaderPLY(file_out, c)

    with open(name_file, "r") as file:
         c = 0
         for x in file:
             if c > 2:
                lst = x.split(" ", 7)
             
                file_out.write(lst[1] + " " + lst[2] + " " + lst[3] + " " +
                               lst[4] + " " + lst[5] + " " + lst[6] + " " + lst[7]"\n")
             c += 1
    file.close()

#
#
#
if __name__ == '__main__':
    n_argv = len(sys.argv)
    
    if n_argv >= 2:
        name_file = sys.argv[1]
    
    writePLY(name_file)
