# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import gc
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)
import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimatorLazy
from tools.vis_utils import visualize_sample_together
from tqdm import tqdm

torch.backends.cuda.preferred_linalg_library("magma")


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def body_loader():
        model, model_cfg = load_sam_3d_body(
            args.checkpoint_path,
            device=device,
            mhr_path=mhr_path,
        )
        return model, model_cfg

    def detector_loader():
        if not args.detector_name:
            return None
        from tools.build_detector import HumanDetector
        return HumanDetector(
            name=args.detector_name,
            device=device,
            path=detector_path,
        )

    def segmentor_loader():
        if not args.segmentor_name:
            return None
        if args.segmentor_name == "sam2" and not len(segmentor_path):
            return None
        from tools.build_sam import HumanSegmentor
        return HumanSegmentor(
            name=args.segmentor_name,
            device=device,
            path=segmentor_path,
        )

    def fov_loader():
        if not args.fov_name:
            return None
        from tools.build_fov_estimator import FOVEstimator
        return FOVEstimator(
            name=args.fov_name,
            device=device,
            path=fov_path,
        )

    estimator = SAM3DBodyEstimatorLazy(
        body_model_loader=body_loader,
        detector_loader=detector_loader,
        segmentor_loader=segmentor_loader,
        fov_estimator_loader=fov_loader,
        device=device,
    )

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp"]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        # Faces are only available after body model is loaded during inference.
        faces = estimator.last_faces
        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, faces)
        cv2.imwrite(
            f"{output_folder}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img.astype(np.uint8),
        )

        _cleanup_cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", required=True, type=str)
    parser.add_argument("--output_folder", default="", type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--detector_name", default="", type=str)
    parser.add_argument("--segmentor_name", default="", type=str)
    parser.add_argument("--fov_name", default="", type=str)
    parser.add_argument("--detector_path", default="", type=str)
    parser.add_argument("--segmentor_path", default="", type=str)
    parser.add_argument("--fov_path", default="", type=str)
    parser.add_argument("--mhr_path", default="", type=str)
    parser.add_argument("--bbox_thresh", default=0.8, type=float)
    parser.add_argument("--use_mask", action="store_true", default=False)
    args = parser.parse_args()
    main(args)