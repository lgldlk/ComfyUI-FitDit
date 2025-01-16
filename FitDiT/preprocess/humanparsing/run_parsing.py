import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference
import torch


class Parsing:
    def __init__(self, model_root, device):
        providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(os.path.join(model_root, 'humanparsing/parsing_atr.onnx'),
                                            sess_options=session_options, providers=providers)
        self.lip_session = ort.InferenceSession(os.path.join(model_root, 'humanparsing/parsing_lip.onnx'),
                                                sess_options=session_options, providers=providers)
        

    def __call__(self, input_image):
#         torch.cuda.set_device(self.gpu_id)
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
