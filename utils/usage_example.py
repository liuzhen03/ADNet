import numpy as np
import cv2
import sys
sys.path.append('.')
import data_io as io
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Training Batch size.")
parser.add_argument("-rp", "--read_path", type=str, default="./", help="Path to the training dataset")
parser.add_argument("-wp", "--write_path", type=str, default="./", help="Path to write output images")
args = parser.parse_args()
read_path = args.read_path
write_path=args.write_path

image_id = 124
# Read ground-truth image
image_gt = io.imread_uint16_png(read_path + "{:04d}_gt.png".format(image_id), read_path + "{:04d}_alignratio.npy".format(image_id))
# Read input triplets
image_short = cv2.cvtColor(cv2.imread(read_path + "{:04d}_short.png".format(image_id), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
image_medium = cv2.cvtColor(cv2.imread(read_path + "{:04d}_medium.png".format(image_id), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
image_long = cv2.cvtColor(cv2.imread(read_path + "{:04d}_long.png".format(image_id), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0

# Read exposures and find relation to the medium frame (i.e. GT aligned to medium frame)
exposures=np.load(read_path + "{:04d}_exposures.npy".format(image_id))
floating_exposures = exposures - exposures[1]

# Canonical EV alignment - please note the camera response function is not known so this is an aproximation based on a gamma step only
# Aproximate gamma
gamma=2.24
image_short_corrected = (((image_short**gamma)*2.0**(-1*floating_exposures[0]))**(1/gamma))
image_long_corrected = (((image_long**gamma)*2.0**(-1*floating_exposures[2]))**(1/gamma))

# Do some processing to the input image(s) to obtain an output image
output_image = image_medium + 0.01
# Write the output image as uint16 png image and its alignratio.npy following the challenge naming convention
io.imwrite_uint16_png(write_path+"{:04d}.png".format(image_id), output_image, write_path+"{:04d}_alignratio.npy".format(image_id))

print("You reached the end of the usage_example.py demo script. Good luck participating in the NTIRE 2021 HDR Challenge!")