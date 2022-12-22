#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import time
from video import *

#
#
#
if __name__ == '__main__':
    n_argv = len(sys.argv)
    
    if n_argv >= 2:
        base_dir = sys.argv[1]
        
        database_str = os.path.join(base_dir, 'database.db')
        print(database_str)

        sparse_str = os.path.join(base_dir, 'sparse')
        print(sparse_str)
        mkdir_s(sparse_str)

        sparse_str0 = os.path.join(base_dir, 'sparse/0')       
        print(sparse_str0)        
        mkdir_s(sparse_str0)

        print('Computing Features...')
        os.system('colmap feature_extractor --database_path ' + database_str + ' --image_path ' + base_dir)

        print('Matching features...')
        os.system('colmap exhaustive_matcher --database_path '+ database_str)
            

        print('Mapping...')
        os.system('colmap mapper --database_path ' + database_str + ' --image_path ' + base_dir + ' --output_path ' + sparse_str)

        print('Converting into text...')
        os.system('colmap model_converter --input_path ' + sparse_str0 + ' --output_path ' + sparse_str + '  --output_type TXT')
        
        print('Storing as a PLY file...')
        name_file_to_convert = os.path.join(sparse_str, 'points3D.txt')
        os.system('python3 cm_convertToPLY.py ' + name_file_to_convert)

