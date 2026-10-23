"""Generate add_one.onnx: action[2, 6] = state + 1 (+ 0 * sum(image))."""

import pathlib

import onnx
from onnx import TensorProto, helper

STATE = "observation.state"
IMAGE = "observation.images.wrist"


def build() -> onnx.ModelProto:
    """Build and validate the add_one model."""
    inputs = [
        helper.make_tensor_value_info(STATE, TensorProto.FLOAT, [6]),
        helper.make_tensor_value_info(IMAGE, TensorProto.UINT8, [4, 4, 3]),
    ]
    outputs = [helper.make_tensor_value_info("action", TensorProto.FLOAT, [2, 6])]
    inits = [
        helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("out_shape", TensorProto.INT64, [2], [2, 6]),
    ]
    nodes = [
        helper.make_node("Add", [STATE, "one"], ["plus_one"]),
        helper.make_node("Expand", ["plus_one", "out_shape"], ["expanded"]),
        helper.make_node("Cast", [IMAGE], ["image_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["image_f"], ["image_sum"], keepdims=0),
        helper.make_node("Mul", ["image_sum", "zero"], ["image_zero"]),
        helper.make_node("Add", ["expanded", "image_zero"], ["action"]),
    ]
    graph = helper.make_graph(nodes, "add_one", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out = pathlib.Path(__file__).with_name("add_one.onnx")
    onnx.save(build(), out)
    print(f"wrote {out}")
