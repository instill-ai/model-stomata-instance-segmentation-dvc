import os
import numpy as np
import json
import cv2
import torch

from typing import List
from itertools import groupby

from triton_python_backend_utils import Tensor, InferenceResponse, \
    get_input_tensor_by_name, InferenceRequest, get_input_config_by_name, \
    get_output_config_by_name, triton_string_to_numpy

classes = [line.rstrip('\n') for line in open(os.path.dirname(__file__) + '/classes.txt')]    


def rle_encode(binary_mask):
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
    uncompressed_rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = uncompressed_rle.get('counts')
    for i, (value, elements) in enumerate(groupby(fortran_binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)    # Add 0 if the mask starts with one, since the odd counts are always the number of zeros
        counts.append(len(list(elements)))
    
    return uncompressed_rle

def post_process(boxes, labels, masks, scores, scale, pad, score_threshold=0.7):
    # Resize boxes
    for b in boxes:
        b[0] -= pad[0]
        b[1] -= pad[1]   
        b[2] -= pad[0]
        b[3] -= pad[1]     
    for b in boxes:
        b[0] /= scale[1]
        b[1] /= scale[0]
        b[2] /= scale[1]
        b[3] /= scale[0]

    rles = []
    ret_boxes = []
    ret_scores = []
    ret_labels = []
    for mask, box, label, score in zip(masks, boxes, labels, scores):
        # Showing boxes with score > 0.7
        if score <= score_threshold:
            continue
        ret_scores.append(score)
        ret_labels.append(classes[int(label)])
        mask = mask[0, :, :, None]
        int_box = [int(i) for i in box]
        mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
        ret_boxes.append([int_box[0], int_box[1], int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1]) # convert to x,y,w,h
        mask = mask > 0.5
        rle = rle_encode(mask).get('counts')
        rle = [str(i) for i in rle]
        rle = ",".join(rle) # output batching need to be same shape then convert rle to string for each object mask
        rles.append(rle)

    return rles, ret_boxes, ret_labels, ret_scores


class TritonPythonModel(object):
    def __init__(self):
        self.input_names = {
            'scale': 'scale',
            'pad': 'pad',
            'boxes': 'boxes',
            'labels': 'labels',
            'scores': 'scores',
            'masks': 'masks',
        }
        self.output_names = {
            'rles': 'rles',
            'boxes': 'boxes',
            'labels': 'labels',
            'scores': 'scores',
        }

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        if 'input' not in model_config:
            raise ValueError('Input is not defined in the model config')

        input_configs = {k: get_input_config_by_name(
            model_config, name) for k, name in self.input_names.items()}
        for k, cfg in input_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Input {self.input_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for input {self.input_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for input {self.input_names[k]} is not defined in the model config')

        if 'output' not in model_config:
            raise ValueError('Output is not defined in the model config')

        output_configs = {k: get_output_config_by_name(
            model_config, name) for k, name in self.output_names.items()}
        for k, cfg in output_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Output {self.output_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for output {self.output_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for output {self.output_names[k]} is not defined in the model config')
            if 'data_type' not in cfg:
                raise ValueError(
                    f'Data type for output {self.output_names[k]} is not defined in the model config')

        self.output_dtypes = {k: triton_string_to_numpy(
            cfg['data_type']) for k, cfg in output_configs.items()}

    def execute(self, inference_requests: List[InferenceRequest]) -> List[InferenceResponse]:
        responses = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for request in inference_requests:
            batch_in = {}
            for k, name in self.input_names.items():
                tensor = get_input_tensor_by_name(request, name)
                if tensor is None:
                    raise ValueError(f'Input tensor {name} not found ' f'in request {request.request_id()}')
                batch_in[k] = tensor.as_numpy()  # shape (batch_size, ...)

            batch_out = {k: [] for k, name in self.output_names.items(
            ) if name in request.requested_output_names()}

            in_scale = batch_in['scale']
            in_pad = batch_in['pad']
            in_boxes = batch_in['boxes'] # batching multiple images
            in_labels = batch_in['labels'] # batching multiple images
            in_scores = batch_in['scores'] # batching multiple images
            in_masks = batch_in['masks'] # batching multiple images

            rs_boxes = []
            rs_labels = []
            rs_rles = []
            rs_scores = []
            for boxes, labels, masks, scores, scale, pad in zip(in_boxes, in_labels, in_masks, in_scores, in_scale, in_pad): # single image
                image_rles, image_boxes, image_labels, image_scores = post_process(boxes, labels, masks, scores, scale, pad)
                rs_boxes.append(image_boxes)
                rs_labels.append(image_labels)
                rs_scores.append(image_scores)
                rs_rles.append(image_rles)

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

            batch_out['boxes'] = rs_boxes
            batch_out['labels'] = rs_labels
            batch_out['scores'] = rs_scores
            batch_out['rles'] = rs_rles

            # Format outputs to build an InferenceResponse
            output_tensors = [Tensor(self.output_names[k], np.asarray(
                out, dtype=self.output_dtypes[k])) for k, out in batch_out.items()]

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses