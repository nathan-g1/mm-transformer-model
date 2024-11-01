'''
Use env gdaltest
'''

import os
from osgeo import gdal
from datetime import datetime
from pathlib import Path

options_list = [
    #'-ot Byte',
    '-of PNG'
    #'-b 1',
    #'-scale'
]

options_string = " ".join(options_list)
tif_dir = "S2_tif"
for i, filename in enumerate(os.listdir(tif_dir)):
    filename_split = filename.split('_')
    img_type = filename_split[len(filename_split) - 1]
    date_string = filename_split[3]
    date = datetime.strptime(date_string, '%Y%m%dT%H%M%S').date()
    new_filename = str(i) + "_" + str(date) + "_" + img_type.replace("tif", "png")

    gdal.Translate(
        os.path.join("png", new_filename),
        os.path.join(tif_dir, filename),
        options=options_string
    )
