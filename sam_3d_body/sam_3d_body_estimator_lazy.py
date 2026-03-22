# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Optional, Union, Callable, Tuple
import gc

import cv2
import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SAM3DBodyEstimatorLazy:
    def __init__(
        self,
        body_model_loader: Callable[[], Tuple[torch.nn.Module, object]],
        detector_loader: Optional[Callable[[], object]] = None,
        segmentor_loader: Optional[Callable[[], object]] = None,
        fov_estimator_loader: Optional[Callable[[], object]] = None,
        device: Union[str, torch.device] = "cuda",
    ):
        self.device = torch.device(device)

        self.body_model_loader = body_model_loader
        self.detector_loader = detector_loader
        self.segmentor_loader = segmentor_loader
        self.fov_estimator_loader = fov_estimator_loader

        self.thresh_wrist_angle = 1.4
        self.last_faces = None

        print("Low-VRAM mode enabled: models will be loaded and unloaded per stage.")

        self.transform = None
        self.transform_hand = None

    def _ensure_transforms(self, cfg):
        if self.transform is None:
            self.transform = Compose(
                [
                    GetBBoxCenterScale(),
                    TopdownAffine(input_size=cfg.MODEL.IMAGE_SIZE, use_udp=False),
                    VisionTransformWrapper(ToTensor()),
                ]
            )

        if self.transform_hand is None:
            self.transform_hand = Compose(
                [
                    GetBBoxCenterScale(padding=0.9),
                    TopdownAffine(input_size=cfg.MODEL.IMAGE_SIZE, use_udp=False),
                    VisionTransformWrapper(ToTensor()),
                ]
            )

    def _load_detector(self):
        if self.detector_loader is None:
            return None
        return self.detector_loader()

    def _load_segmentor(self):
        if self.segmentor_loader is None:
            return None
        return self.segmentor_loader()

    def _load_fov(self):
        if self.fov_estimator_loader is None:
            return None
        return self.fov_estimator_loader()

    def _load_body_model(self):
        model, cfg = self.body_model_loader()
        self._ensure_transforms(cfg)
        self.last_faces = model.head_pose.faces.cpu().numpy()
        return model, cfg
    def _stage(self, msg: str) -> None:
        print(f"[sam3d] {msg}")
    @torch.inference_mode()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        self._stage("start")

        if type(img) == str:
            self._stage(f"load image: {img}")
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            self._stage("input image provided as ndarray")
            print("####### Please make sure the input image is in RGB format")
            image_format = "rgb"

        height, width = img.shape[:2]

        # Stage 1: detection
        if bboxes is not None:
            self._stage("detector: skipped (using provided bboxes)")
            boxes = bboxes.reshape(-1, 4)
            is_crop = True
        else:
            detector = self._load_detector()
            if detector is not None:
                self._stage("detector: run")
                if image_format == "rgb":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    image_format = "bgr"

                boxes = detector.run_human_detection(
                    img,
                    det_cat_id=det_cat_id,
                    bbox_thr=bbox_thr,
                    nms_thr=nms_thr,
                    default_to_full_image=False,
                )
                self._stage(f"detector: found {len(boxes)} boxes")

                del detector
                _cleanup_cuda()
                self._stage("detector: unloaded")
                is_crop = True
            else:
                self._stage("detector: skipped (full image fallback)")
                boxes = np.array([0, 0, width, height]).reshape(1, 4)
                is_crop = False

        self.is_crop = is_crop

        if len(boxes) == 0:
            self._stage("stop: no boxes found")
            return []

        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Stage 2: segmentation
        masks_score = None
        if masks is not None:
            self._stage("segmentor: skipped (using provided masks)")
            assert bboxes is not None, "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(len(masks), dtype=np.float32)
            use_mask = True
        elif use_mask:
            segmentor = self._load_segmentor()
            if segmentor is None:
                self._stage("segmentor: requested but unavailable")
                masks, masks_score = None, None
            else:
                self._stage("segmentor: run")
                masks, masks_score = segmentor.run_sam(img, boxes)
                self._stage(f"segmentor: produced {len(masks) if masks is not None else 0} masks")
                del segmentor
                _cleanup_cuda()
                self._stage("segmentor: unloaded")
        else:
            self._stage("segmentor: skipped (use_mask=False)")
            masks, masks_score = None, None

        # Stage 3: body model load
        self._stage("body model: load")
        model, cfg = self._load_body_model()
        self._stage("body model: loaded")

        batch = prepare_batch(img, self.transform, boxes, masks, masks_score)
        batch = recursive_to(batch, self.device)
        model._initialize_batch(batch)

        # Stage 4: FOV
        if cam_int is not None:
            self._stage("fov: skipped (using provided cam_int)")
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        else:
            fov_estimator = self._load_fov()
            if fov_estimator is not None:
                self._stage("fov: run")
                input_image = batch["img_ori"][0].data
                cam_int = fov_estimator.get_cam_intrinsics(input_image).to(batch["img"])
                batch["cam_int"] = cam_int.clone()
                del fov_estimator
                _cleanup_cuda()
                self._stage("fov: unloaded")
            else:
                self._stage("fov: skipped (default intrinsics)")
                cam_int = batch["cam_int"].clone()

        # Stage 5: inference
        self._stage(f"body model: inference ({inference_type})")
        outputs = model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        self._stage("body model: inference done")

        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs
            batch_lhand, batch_rhand = None, None

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")

        all_out = []
        for idx in range(batch["img"].shape[1]):
            item = {
                "bbox": batch["bbox"][0, idx].cpu().numpy(),
                "focal_length": out["focal_length"][idx],
                "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                "pred_vertices": out["pred_vertices"][idx],
                "pred_cam_t": out["pred_cam_t"][idx],
                "pred_pose_raw": out["pred_pose_raw"][idx],
                "global_rot": out["global_rot"][idx],
                "body_pose_params": out["body_pose"][idx],
                "hand_pose_params": out["hand"][idx],
                "scale_params": out["scale"][idx],
                "shape_params": out["shape"][idx],
                "expr_params": out["face"][idx],
                "mask": masks[idx] if masks is not None else None,
                "pred_joint_coords": out["pred_joint_coords"][idx],
                "pred_global_rots": out["joint_global_rots"][idx],
                "mhr_model_params": out["mhr_model_params"][idx],
            }

            if inference_type == "full" and batch_lhand is not None and batch_rhand is not None:
                item["lhand_bbox"] = np.array(
                    [
                        (batch_lhand["bbox_center"].flatten(0, 1)[idx][0] - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2).item(),
                        (batch_lhand["bbox_center"].flatten(0, 1)[idx][1] - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2).item(),
                        (batch_lhand["bbox_center"].flatten(0, 1)[idx][0] + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2).item(),
                        (batch_lhand["bbox_center"].flatten(0, 1)[idx][1] + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2).item(),
                    ]
                )
                item["rhand_bbox"] = np.array(
                    [
                        (batch_rhand["bbox_center"].flatten(0, 1)[idx][0] - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2).item(),
                        (batch_rhand["bbox_center"].flatten(0, 1)[idx][1] - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2).item(),
                        (batch_rhand["bbox_center"].flatten(0, 1)[idx][0] + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2).item(),
                        (batch_rhand["bbox_center"].flatten(0, 1)[idx][1] + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2).item(),
                    ]
                )

            all_out.append(item)

        del outputs, pose_output, out, model, batch
        if batch_lhand is not None:
            del batch_lhand
        if batch_rhand is not None:
            del batch_rhand
        _cleanup_cuda()
        self._stage("body model: unloaded")
        self._stage("done")
        return all_out