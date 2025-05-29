import tensorflow as tf
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="/home/wld-algo-6/Production/WeightEstimation2K/logs/29-05-2025-10-58-35/snc_weight_estimation_model.tflite")
interpreter.allocate_tensors()

# Get the signature runner
signatures = interpreter.get_signature_list()
if "my_signature" in signatures:
    signature_runner = interpreter.get_signature_runner("my_signature")
else:
    print("Signature 'my_signature' not found.")
    exit()

# Prepare the input data
input_shape = interpreter.get_input_details()[0]['shape']
input_data = np.random.rand(*input_shape).astype(np.float32)
input_tensor_name = "input_tensor_name"

inputs = {input_tensor_name: input_data}

# Run inference
output_data = signature_runner(**inputs)

# Retrieve the output
output_tensor_name = "output_tensor_name"
output = output_data[output_tensor_name]

print(output)