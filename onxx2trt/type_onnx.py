import onnx
from onnx import helper

# Load the ONNX model
model_path = "albert-base-v2_mbti-classification.onnx"
model = onnx.load(model_path)

# Function to print tensor type information
def print_tensor_type(tensor):
    # Get the type of the tensor
    tensor_type = tensor.type.tensor_type
    # Get the element type
    elem_type = tensor_type.elem_type
    # Use helper.tensor_dtype_to_np_dtype to get the numpy dtype
    np_dtype = helper.tensor_dtype_to_np_dtype(elem_type)
    # Get the dtype name in a compatible way
    type_str = np_dtype.name
    # Get the shape of the tensor
    shape = [dim.dim_value for dim in tensor_type.shape.dim]
    print(f"Type: {type_str}, Shape: {shape}")

# Print input names and types
print("Model Inputs:")
for input_tensor in model.graph.input:
    print(f"Name: {input_tensor.name}")
    print_tensor_type(input_tensor)

# Print output names and types
print("\nModel Outputs:")
for output_tensor in model.graph.output:
    print(f"Name: {output_tensor.name}")
    print_tensor_type(output_tensor)