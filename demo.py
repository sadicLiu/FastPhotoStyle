from __future__ import print_function

import argparse

import torch
import os

import process_stylization
from photo_wct import PhotoWCT

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth')
parser.add_argument('--content_image_path', default='./images/tank')
parser.add_argument('--content_seg_path', default=[])
parser.add_argument('--style_image_path', default='./images/style_sand2.jpg')
parser.add_argument('--style_seg_path', default=[])
parser.add_argument('--output_image_path', default='./results_canon_sand/')
parser.add_argument('--save_intermediate', action='store_true', default=True)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=True)
parser.add_argument('--cuda', type=int, default=0, help='Enable CUDA.')
args = parser.parse_args()

# stylization module
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))

# smoothing module
if args.fast:
    from photo_gif import GIFSmoothing

    p_pro = GIFSmoothing(r=35, eps=0.001)
else:
    from photo_smooth import Propagator

    p_pro = Propagator()

# process_stylization.stylization(
#     stylization_module=p_wct,
#     smoothing_module=p_pro,
#     content_image_path=args.content_image_path,
#     style_image_path=args.style_image_path,
#     content_seg_path=args.content_seg_path,
#     style_seg_path=args.style_seg_path,
#     output_image_path=args.output_image_path,
#     cuda=args.cuda,
#     save_intermediate=args.save_intermediate,
#     no_post=args.no_post
# )

img_path = './images/canon/'
imgs = os.listdir(img_path)
for a_img in imgs:
    content_img = os.path.join(img_path, a_img)

    process_stylization.stylization(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        # content_image_path=args.content_image_path,
        content_image_path=content_img,
        style_image_path=args.style_image_path,
        content_seg_path=args.content_seg_path,
        style_seg_path=args.style_seg_path,
        output_image_path=args.output_image_path,
        cuda=args.cuda,
        save_intermediate=args.save_intermediate,
        no_post=args.no_post
    )
