#!/usr/bin/env python
import sys
import os.path
import numpy as np
import data_io as io
from metrics import normalized_psnr, psnr_tanh_norm_mu_tonemap
# as per the metadata file, input and output directories are the arguments
[_, input_dir, output_dir] = sys.argv

res_dir = os.path.join(input_dir, 'res/')
ref_dir = os.path.join(input_dir, 'ref/')
#print("REF DIR")
#print(ref_dir)


runtime = -1
cpu = -1
data = -1
other = ""
readme_fnames = [p for p in os.listdir(res_dir) if p.lower().startswith('readme')]
try:
    readme_fname = readme_fnames[0]
    print("Parsing extra information from %s"%readme_fname)
    with open(os.path.join(input_dir, 'res', readme_fname)) as readme_file:
        readme = readme_file.readlines()
        lines = [l.strip() for l in readme if l.find(":")>=0]
        runtime = float(":".join(lines[0].split(":")[1:]))
        cpu = int(":".join(lines[1].split(":")[1:]))
        data = int(":".join(lines[2].split(":")[1:]))
        other = ":".join(lines[3].split(":")[1:])
except:
    print("Error occured while parsing readme.txt")
    print("Please make sure you have a line for runtime, cpu/gpu, extra data and other (4 lines in total).")
print("Parsed information:")
print("Runtime: %f"%runtime)
print("CPU/GPU: %d"%cpu)
print("Data: %d"%data)
print("Other: %s"%other)





ref_pngs = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('.png')])
ref_alignratios = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('.npy')])

res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])
res_alignratios = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('npy')])

if not (len(ref_pngs)==len(res_pngs)==len(ref_alignratios)==len(res_alignratios)):
    raise Exception('Expected %d .png images'%len(ref_pngs))




scores = []
scores_mupsnr = []
for (ref_im, ref_alignratio, res_im, res_alignratio) in zip(ref_pngs, ref_alignratios, res_pngs, res_alignratios):
    print(ref_im, ref_alignratio, res_im, res_alignratio)
    # Read images
    ref_hdr_image = io.imread_uint16_png(os.path.join(input_dir, 'ref', ref_im), os.path.join(input_dir, 'ref', ref_alignratio))
    res_hdr_image = io.imread_uint16_png(os.path.join(input_dir, 'res', res_im), os.path.join(input_dir, 'res', res_alignratio))
    # normalized_psnr(ref_hdr_image, res_hdr_image, np.max(ref_hdr_image))
    scores.append(
        normalized_psnr(ref_hdr_image, res_hdr_image, np.max(ref_hdr_image))
    )
    scores_mupsnr.append(
        psnr_tanh_norm_mu_tonemap(ref_hdr_image, res_hdr_image, percentile=99, gamma=2.24)
    )

    #print(scores[-1])
psnr = np.mean(scores)
mu_psnr = np.mean(scores_mupsnr)



# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    output_file.write("PSNR:%f\n"%psnr)
    output_file.write("MuPSNR:%f\n"%mu_psnr)
    output_file.write("ExtraRuntime:%f\n"%runtime)
    output_file.write("ExtraPlatform:%d\n"%cpu)
    output_file.write("ExtraData:%d\n"%data)
