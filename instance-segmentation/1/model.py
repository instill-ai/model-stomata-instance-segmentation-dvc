import os
from itertools import groupby

import ray
import torch
import cv2
import numpy as np

from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.augmentations import letterbox
from utils.segment.general import process_mask, scale_masks
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from ray import serve
from instill.helpers.const import DataType
from instill.helpers.ray_io import serialize_byte_tensor, deserialize_bytes_tensor
from instill.helpers.ray_config import (
    InstillRayModelConfig,
    entry,
)

from ray_pb2 import (
    ModelReadyRequest,
    ModelReadyResponse,
    ModelMetadataRequest,
    ModelMetadataResponse,
    ModelInferRequest,
    ModelInferResponse,
    InferTensor,
)


@serve.deployment()
class StomataYolov7:
    def __init__(self, model_path: str):
        self.device = select_device("cuda:0")
        self.model = DetectMultiBackend(
            model_path, device=self.device, dnn=False, data=None, fp16=True
        )

        self.image_size = check_img_size(640, s=self.model.stride)

        self.model.warmup()  # warmup

    def ModelMetadata(self, req: ModelMetadataRequest) -> ModelMetadataResponse:
        resp = ModelMetadataResponse(
            name=req.name,
            versions=req.version,
            framework="pytorch",
            inputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="input",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="rles",
                    shape=[-1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="boxes",
                    shape=[-1, 4],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="labels",
                    shape=[-1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="scores",
                    shape=[-1],
                ),
            ],
        )
        return resp

    def ModelReady(self, req: ModelReadyRequest) -> ModelReadyResponse:
        resp = ModelReadyResponse(ready=True)
        return resp

    def rle_encode(self, binary_mask):
        r"""
        Args:
            binary_mask: a binary mask with the shape of `mask_shape`

        Returns uncompressed Run-length Encoding (RLE) in COCO format
                Link: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
                {
                    'counts': [n1, n2, n3, ...],
                    'size': [height, width] of the mask
                }
        """
        fortran_binary_mask = np.asfortranarray(binary_mask)
        uncompressed_rle = {"counts": [], "size": list(binary_mask.shape)}
        counts = uncompressed_rle.get("counts")
        for i, (value, elements) in enumerate(
            groupby(fortran_binary_mask.ravel(order="F"))
        ):
            if i == 0 and value == 1:
                counts.append(
                    0
                )  # Add 0 if the mask starts with one, since the odd counts are always the number of zeros
            counts.append(len(list(elements)))

        return uncompressed_rle

    def post_process(self, boxes, labels, masks, scores, score_threshold=0.7):
        rles = []
        ret_boxes = []
        ret_scores = []
        ret_labels = []
        for mask, box, label, score in zip(masks, boxes, labels, scores):
            box = box.cpu()
            score = score.cpu()
            # Showing boxes with score > 0.7
            if score <= score_threshold:
                continue
            ret_scores.append(score)
            ret_labels.append(label)
            int_box = [int(i) for i in box]
            mask = mask[int_box[1] : int_box[3] + 1, int_box[0] : int_box[2] + 1]
            ret_boxes.append(
                [
                    int_box[0],
                    int_box[1],
                    int_box[2] - int_box[0] + 1,
                    int_box[3] - int_box[1] + 1,
                ]
            )  # convert to x,y,w,h
            mask = mask > 0.5
            rle = self.rle_encode(mask).get("counts")
            rle = [str(i) for i in rle]
            rle = ",".join(
                rle
            )  # output batching need to be same shape then convert rle to string for each object mask
            rles.append(rle)

        return rles, ret_boxes, ret_labels, ret_scores

    async def ModelInfer(self, request: ModelInferRequest) -> ModelInferResponse:
        resp = ModelInferResponse(
            model_name=request.model_name,
            model_version=request.model_version,
            outputs=[],
            raw_output_contents=[],
        )

        b_tensors = request.raw_input_contents[0]

        input_tensors = deserialize_bytes_tensor(b_tensors)

        image_masks = []
        image_boxes = []
        image_scores = []
        image_labels = []
        for enc in input_tensors:
            frame = cv2.imdecode(np.frombuffer(enc, np.uint8), cv2.IMREAD_COLOR)
            im = letterbox(frame, self.image_size, stride=self.model.stride)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred, out = self.model(im)
            proto = out[1]
            pred = non_max_suppression(
                pred,
                conf_thres=0.3,
                max_det=1000,
                nm=32,
            )

            det_masks = []
            for i, det in enumerate(pred):  # per image
                if len(det):  # per detection
                    masks = process_mask(
                        proto[i], det[:, 6:], det[:, :4], im.shape[2:], True
                    )  # HWC
                    masks = scale_masks(
                        im.shape[2:],
                        masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                        frame.shape,
                    )
                    masks = np.transpose(masks, (2, 0, 1))
                    det[:, :4] = scale_coords(
                        im.shape[2:], det[:, :4], frame.shape
                    ).round()

                    det_masks.extend(masks)

            image_masks.append(det_masks)
            image_boxes.append(pred[0][:, :4])
            image_scores.append(pred[0][:, 4])
            image_labels.append(["stomata"] * len(pred[0]))

        # og post process
        rs_boxes = []
        rs_labels = []
        rs_rles = []
        rs_scores = []

        for boxes, labels, masks, scores in zip(
            image_boxes, image_labels, image_masks, image_scores
        ):  # single image
            o_rles, o_boxes, o_labels, o_scores = self.post_process(
                boxes, labels, masks, scores
            )
            rs_boxes.append(o_boxes)
            rs_labels.append(o_labels)
            rs_scores.append(o_scores)
            rs_rles.append(o_rles)

        max_boxes = max([len(i) for i in rs_boxes])
        for b in rs_boxes:
            for _ in range(max_boxes - len(b)):
                b.append([0, 0, 0, 0])
        max_labels = max([len(i) for i in rs_labels])
        for lb in rs_labels:
            for _ in range(max_labels - len(lb)):
                lb.append("")
        max_scores = max([len(i) for i in rs_scores])
        for sc in rs_scores:
            for _ in range(max_scores - len(sc)):
                sc.append(0)

        max_rles = max([len(i) for i in rs_rles])
        for rles in rs_rles:
            for _ in range(max_rles - len(rles)):
                rles.append("")

        # rles
        resp.outputs.append(
            InferTensor(
                name="rles",
                shape=[len(input_tensors), len(rs_rles[0])],
                datatype=str(DataType.TYPE_STRING),
            )
        )
        rles_out = []
        for r in rs_rles:
            rles_out.extend(r)
        if len(rles_out) != 0:
            rles_out = [bytes(f"{rles_out[i]}", "utf-8") for i in range(len(rles_out))]
            resp.raw_output_contents.append(serialize_byte_tensor(np.asarray(rles_out)))
        else:
            resp.raw_output_contents.append(b"")

        # boxes
        resp.outputs.append(
            InferTensor(
                name="boxes",
                shape=[len(input_tensors), len(rs_boxes[0]), 4],
                datatype=str(DataType.TYPE_FP32),
            )
        )
        resp.raw_output_contents.append(
            np.asarray(rs_boxes).astype(np.float32).tobytes()
        )

        # labels
        resp.outputs.append(
            InferTensor(
                name="labels",
                shape=[len(input_tensors), len(rs_labels[0])],
                datatype=str(DataType.TYPE_STRING),
            )
        )
        labels_out = []
        for r in rs_labels:
            labels_out.extend(r)
        if len(labels_out) != 0:
            labels_out = [
                bytes(f"{labels_out[i]}", "utf-8") for i in range(len(labels_out))
            ]
            resp.raw_output_contents.append(
                serialize_byte_tensor(np.asarray(labels_out))
            )
        else:
            resp.raw_output_contents.append(b"")

        # scores
        resp.outputs.append(
            InferTensor(
                name="scores",
                shape=[len(input_tensors), len(rs_scores[0])],
                datatype=str(DataType.TYPE_FP32),
            )
        )
        resp.raw_output_contents.append(
            np.asarray(rs_scores).astype(np.float32).tobytes()
        )

        return resp


def deploy_model(model_config: InstillRayModelConfig):
    c_app = StomataYolov7.options(
        name=model_config.application_name,
        ray_actor_options=model_config.ray_actor_options,
        max_concurrent_queries=model_config.max_concurrent_queries,
        autoscaling_config=model_config.ray_autoscaling_options,
    ).bind(model_config.model_path)

    serve.run(
        c_app, name=model_config.model_name, route_prefix=model_config.route_prefix
    )


def undeploy_model(model_name: str):
    serve.delete(model_name)


if __name__ == "__main__":
    func, model_config = entry("model.pt")

    ray.init(
        address=model_config.ray_addr,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": os.getcwd(),
            },
        },
    )

    model_config.ray_actor_options["num_cpus"] = 0.4
    model_config.ray_actor_options["num_gpus"] = 0.1

    if func == "deploy":
        deploy_model(model_config=model_config)
    elif func == "undeploy":
        undeploy_model(model_name=model_config.model_name)
