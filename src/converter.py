from osgeo import gdal

options_list = [
    #'-ot Byte',
    '-of PNG'
    #'-b 1',
    #'-scale'
]           

options_string = " ".join(options_list)

'''
for i in range(33,65):
    input = 's2_vis_'+ str(i) +'.tif'
    output = 's2_vis_'+ str(i) +'.png'
    #print(input)
    #print(output)
    gdal.Translate(
        output,
        input,
        options=options_string
    )
'''

gdal.Translate(
    's2_vis_58_new.png',
    's2_vis_58_new.tif',
    options=options_string
)