import onnx
from onnxsim import simplify
 
ONNX_MODEL_PATH = '../yolov5-opencv-cpp-python/config_files/middle.onnx'
ONNX_SIM_MODEL_PATH = '../yolov5-opencv-cpp-python/config_files/middle_sim.onnx'
 
if __name__ == "__main__":
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx_sim_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_sim_model, ONNX_SIM_MODEL_PATH)
    print('ONNX file simplified!')

