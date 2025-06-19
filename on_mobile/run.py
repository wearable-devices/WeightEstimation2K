

import tensorflow as tf
import numpy as np
import os


# def check_tflite_model(model_path):
#     """
#     Simple TFLite model checker based on your original code
#     """
#     print(f"🔍 Checking TFLite model: {os.path.basename(model_path)}")
#
#     # Check if file exists
#     if not os.path.exists(model_path):
#         print(f"❌ Model file not found: {model_path}")
#         return False
#
#     try:
#         # Load the TFLite model
#         print("📖 Loading TFLite interpreter...")
#         interpreter = tf.lite.Interpreter(model_path=model_path)
#         interpreter.allocate_tensors()
#         print("✅ Model loaded successfully!")
#
#     except Exception as e:
#         print(f"❌ Failed to load model: {e}")
#         return False
#
#     # Get model information
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     signatures = interpreter.get_signature_list()
#
#     print(f"\n📊 Model Information:")
#     print(f"   Signatures: {list(signatures.keys())}")
#     print(f"   Inputs: {len(input_details)}")
#     print(f"   Outputs: {len(output_details)}")
#
#     # Print input details
#     print(f"\n📥 Input Details:")
#     for i, detail in enumerate(input_details):
#         print(f"   Input {i}: '{detail['name']}' - Shape: {detail['shape']} - Type: {detail['dtype']}")
#
#     # Print output details
#     print(f"\n📤 Output Details:")
#     for i, detail in enumerate(output_details):
#         print(f"   Output {i}: '{detail['name']}' - Shape: {detail['shape']} - Type: {detail['dtype']}")
#
#     # Method 1: Test with signatures (if available)
#     if signatures:
#         print(f"\n🧪 Testing with signatures...")
#
#         for signature_name in signatures.keys():
#             print(f"   Testing signature: '{signature_name}'")
#
#             try:
#                 signature_runner = interpreter.get_signature_runner(signature_name)
#
#                 # Create test inputs based on input details
#                 test_inputs = {}
#                 for detail in input_details:
#                     input_name = detail['name']
#                     input_shape = detail['shape']
#                     input_dtype = detail['dtype']
#
#                     # Handle dynamic batch size
#                     actual_shape = []
#                     for dim in input_shape:
#                         actual_shape.append(1 if dim == -1 else dim)
#
#                     # Create test data (you can modify this part for your specific data)
#                     if 'snc' in input_name.lower():
#                         # Sensor data - use realistic values
#                         test_data = np.random.normal(0, 0.5, actual_shape).astype(input_dtype)#.type)
#                     else:
#                         # General test data
#                         test_data = np.random.random(actual_shape).astype(input_dtype.type)
#
#                     test_inputs[input_name] = test_data
#                     print(f"      Created input '{input_name}': shape {actual_shape}")
#
#                 # Run inference with signature
#                 print(f"      Running inference...")
#                 output_data = signature_runner(**test_inputs) # [val for val in test_inputs.values()]
#
#                 print(f"   ✅ Signature '{signature_name}' successful!")
#                 print(f"      Output keys: {list(output_data.keys())}")
#
#                 # Print output values
#                 for output_name, output_value in output_data.items():
#                     print(f"      '{output_name}': {output_value} (shape: {output_value.shape})")
#
#                 return True  # Success!
#
#             except Exception as e:
#                 print(f"   ❌ Signature '{signature_name}' failed: {e}")
#
#     # Method 2: Test with manual tensor setting (fallback)
#     print(f"\n🔧 Testing with manual tensor setting...")
#
#     try:
#         # Create and set test inputs
#         for detail in input_details:
#             input_shape = detail['shape']
#             input_dtype = detail['dtype']
#             input_index = detail['index']
#
#             # Handle dynamic batch size
#             actual_shape = []
#             for dim in input_shape:
#                 actual_shape.append(1 if dim == -1 else dim)
#
#             # Create test data
#             if len(actual_shape) == 2 and actual_shape[1] == 1394:
#                 # Looks like sensor data
#                 test_data = np.random.normal(0, 0.1, actual_shape).astype(input_dtype.type)
#             else:
#                 test_data = np.random.random(actual_shape).astype(input_dtype.type)
#
#             interpreter.set_tensor(input_index, test_data)
#             print(f"   Set input {input_index}: shape {actual_shape}")
#
#         # Run inference
#         print("   Running inference...")
#         interpreter.invoke()
#
#         # Get outputs
#         for detail in output_details:
#             output_data = interpreter.get_tensor(detail['index'])
#             print(f"   Output '{detail['name']}': {output_data} (shape: {output_data.shape})")
#
#         print("   ✅ Manual tensor method successful!")
#         return True
#
#     except Exception as e:
#         print(f"   ❌ Manual tensor method failed: {e}")
#         return False

def inspect_signature_details(interpreter):
    """
    Inspect signature details to understand expected inputs
    """
    signatures = interpreter.get_signature_list()

    print("🔍 Signature Analysis:")

    for sig_name, signature in signatures.items():
        print(f"\n📋 Signature: '{sig_name}'")

        # Get signature runner to inspect its structure
        try:
            signature_runner = interpreter.get_signature_runner(sig_name)

            # Try to access signature details through the interpreter
            signature_def = interpreter._get_signature_def(sig_name)

            print(f"   Expected inputs: {len(signature_def.inputs)}")
            for input_key, input_info in signature_def.inputs.items():
                print(f"     '{input_key}': tensor index {input_info.tensor_index}")

            print(f"   Expected outputs: {len(signature_def.outputs)}")
            for output_key, output_info in signature_def.outputs.items():
                print(f"     '{output_key}': tensor index {output_info.tensor_index}")

        except Exception as e:
            print(f"   Could not inspect signature details: {e}")


# def check_tflite_model_fixed(model_path):
#     """
#     Fixed TFLite model checker that handles signature input mismatches
#     """
#     print(f"🔍 Checking TFLite model: {os.path.basename(model_path)}")
#
#     if not os.path.exists(model_path):
#         print(f"❌ Model file not found: {model_path}")
#         return False
#
#     try:
#         # Load the TFLite model
#         interpreter = tf.lite.Interpreter(model_path=model_path)
#         interpreter.allocate_tensors()
#         print("✅ Model loaded successfully!")
#
#     except Exception as e:
#         print(f"❌ Failed to load model: {e}")
#         return False
#
#     # Get model information
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     signatures = interpreter.get_signature_list()
#
#     print(f"\n📊 Model Structure:")
#     print(f"   Signatures: {list(signatures.keys())}")
#     print(f"   Total inputs: {len(input_details)}")
#     print(f"   Total outputs: {len(output_details)}")
#
#     # Detailed input analysis
#     print(f"\n📥 Input Details:")
#     for i, detail in enumerate(input_details):
#         shape_str = str(detail['shape']).replace('-1', 'batch')
#         print(f"   Input {i}: '{detail['name']}' - Shape: {shape_str} - Type: {detail['dtype']}")
#
#     print(f"\n📤 Output Details:")
#     for i, detail in enumerate(output_details):
#         shape_str = str(detail['shape']).replace('-1', 'batch')
#         print(f"   Output {i}: '{detail['name']}' - Shape: {shape_str} - Type: {detail['dtype']}")
#
#     # Inspect signatures in detail
#     if signatures:
#         inspect_signature_details(interpreter)
#
#     # Strategy 1: Test signatures with proper input mapping
#     if signatures:
#         print(f"\n🧪 Testing Signatures (Fixed approach)...")
#
#         for sig_name in signatures.keys():
#             print(f"\n   Testing signature: '{sig_name}'")
#
#             try:
#                 signature_runner = interpreter.get_signature_runner(sig_name)
#
#                 # Method 1a: Try with single combined input (most likely case)
#                 print("      Trying single combined input...")
#
#                 # Find the largest input (likely the combined one)
#                 largest_input = max(input_details, key=lambda x: np.prod(x['shape'][1:]) if len(x['shape']) > 1 else 0)
#
#                 single_input_shape = largest_input['shape']
#                 actual_shape = [1 if dim == -1 else dim for dim in single_input_shape]
#
#                 # Create single test input
#                 if len(actual_shape) >= 2 and actual_shape[-1] == 3:
#                     # Looks like stacked SNC data: (batch, features, 3)
#                     test_data = np.random.normal(0, 0.1, actual_shape).astype(largest_input['dtype'].type)
#                 else:
#                     test_data = np.random.normal(0, 0.1, actual_shape).astype(largest_input['dtype'].type)
#
#                 # Try running with single input
#                 single_input = {largest_input['name']: test_data}
#                 print(f"         Input: '{largest_input['name']}' with shape {actual_shape}")
#
#                 output_data = signature_runner(**single_input)
#
#                 print(f"      ✅ Single input method successful!")
#                 print(f"         Output keys: {list(output_data.keys())}")
#
#                 for output_name, output_value in output_data.items():
#                     if hasattr(output_value, 'shape'):
#                         print(f"         '{output_name}': {output_value.flatten()} (shape: {output_value.shape})")
#                     else:
#                         print(f"         '{output_name}': {output_value}")
#
#                 return True, 'single_input', single_input
#
#             except Exception as e1:
#                 print(f"      ❌ Single input failed: {e1}")
#
#                 # Method 1b: Try with first input only
#                 try:
#                     print("      Trying first input only...")
#
#                     first_input = input_details[0]
#                     actual_shape = [1 if dim == -1 else dim for dim in first_input['shape']]
#                     test_data = np.random.normal(0, 0.1, actual_shape).astype(first_input['dtype'].type)
#
#                     first_input_only = {first_input['name']: test_data}
#                     print(f"         Input: '{first_input['name']}' with shape {actual_shape}")
#
#                     output_data = signature_runner(**first_input_only)
#
#                     print(f"      ✅ First input only method successful!")
#                     print(f"         Output keys: {list(output_data.keys())}")
#
#                     for output_name, output_value in output_data.items():
#                         if hasattr(output_value, 'shape'):
#                             print(f"         '{output_name}': {output_value.flatten()} (shape: {output_value.shape})")
#                         else:
#                             print(f"         '{output_name}': {output_value}")
#
#                     return True, 'first_input_only', first_input_only
#
#                 except Exception as e2:
#                     print(f"      ❌ First input only failed: {e2}")
#
#     # Strategy 2: Manual tensor setting (always works if model loads)
#     print(f"\n🔧 Testing Manual Tensor Setting...")
#
#     try:
#         # Create test data for all inputs
#         for detail in input_details:
#             actual_shape = [1 if dim == -1 else dim for dim in detail['shape']]
#
#             if 'snc' in detail['name'].lower() or len(actual_shape) >= 2:
#                 test_data = np.random.normal(0, 0.1, actual_shape).astype(detail['dtype'].type)
#             else:
#                 test_data = np.random.random(actual_shape).astype(detail['dtype'].type)
#
#             interpreter.set_tensor(detail['index'], test_data)
#             print(f"   Set input '{detail['name']}': shape {actual_shape}")
#
#         # Run inference
#         interpreter.invoke()
#
#         # Get outputs
#         results = {}
#         for detail in output_details:
#             output_data = interpreter.get_tensor(detail['index'])
#             results[detail['name']] = output_data
#             if hasattr(output_data, 'shape'):
#                 print(f"   Output '{detail['name']}': {output_data.flatten()} (shape: {output_data.shape})")
#             else:
#                 print(f"   Output '{detail['name']}': {output_data}")
#
#         print("   ✅ Manual tensor method successful!")
#         return True, 'manual_tensors', results
#
#     except Exception as e:
#         print(f"   ❌ Manual tensor method failed: {e}")
#         return False, None, None

def test_signature_strategies(interpreter, sig_name, input_details):
    """
    Test different strategies for signature inputs without using private methods
    """
    try:
        signature_runner = interpreter.get_signature_runner(sig_name)
    except Exception as e:
        print(f"      ❌ Could not get signature runner: {e}")
        return False, None, None

    # Strategy 1: Single input (most common case for converted models)
    print("      Strategy 1: Single input...")
    try:
        single_input = {}
        for i in range(len(input_details)):
            # Use the first/largest input
            main_input = input_details[i]

            actual_shape = [1 if dim == -1 else dim for dim in main_input['shape']]
            test_data = np.random.normal(0, 0.1, actual_shape).astype(main_input['dtype'])#.type)

            single_input[main_input['name']]= test_data
            print(f"         Trying input: '{main_input['name']}' with shape {actual_shape}")

        output_data = signature_runner(**{'snc_2':test_data, 'snc_1':test_data, 'snc_3':test_data})

        print(f"         ✅ Single input successful!")
        for output_name, output_value in output_data.items():
            if hasattr(output_value, 'shape') and output_value.size > 1:
                print(
                    f"            '{output_name}': shape {output_value.shape}, values {output_value.flatten()[:3]}...")
            else:
                print(f"            '{output_name}': {output_value}")

        return True, 'single_input', single_input

    except Exception as e:
        print(f"         ❌ Single input failed: {e}")

    # Strategy 2: All inputs with individual names
    print("      Strategy 2: All individual inputs...")
    try:
        all_inputs = {}
        for detail in input_details:
            actual_shape = [1 if dim == -1 else dim for dim in detail['shape']]
            test_data = np.random.normal(0, 0.1, actual_shape).astype(detail['dtype'])#.type)
            all_inputs[detail['name']] = test_data
            print(f"         Adding input: '{detail['name']}' with shape {actual_shape}")

        output_data = signature_runner(**all_inputs)

        print(f"         ✅ All inputs successful!")
        for output_name, output_value in output_data.items():
            if hasattr(output_value, 'shape') and output_value.size > 1:
                print(
                    f"            '{output_name}': shape {output_value.shape}, values {output_value.flatten()[:3]}...")
            else:
                print(f"            '{output_name}': {output_value}")

        return True, 'all_inputs', all_inputs

    except Exception as e:
        print(f"         ❌ All inputs failed: {e}")

    # Strategy 3: Try with generic input names
    print("      Strategy 3: Generic input names...")
    try:
        # Common generic names that TFLite conversion might use
        generic_names = ['input', 'input_1', 'x', 'inputs', 'serving_default_input']

        for generic_name in generic_names:
            try:
                main_input = input_details[0]
                actual_shape = [1 if dim == -1 else dim for dim in main_input['shape']]
                test_data = np.random.normal(0, 0.1, actual_shape).astype(main_input['dtype'].type)

                generic_input = {generic_name: test_data}
                print(f"         Trying generic name: '{generic_name}'")

                output_data = signature_runner(**generic_input)

                print(f"         ✅ Generic name '{generic_name}' successful!")
                for output_name, output_value in output_data.items():
                    if hasattr(output_value, 'shape') and output_value.size > 1:
                        print(
                            f"            '{output_name}': shape {output_value.shape}, values {output_value.flatten()[:3]}...")
                    else:
                        print(f"            '{output_name}': {output_value}")

                return True, f'generic_name_{generic_name}', generic_input

            except:
                continue

        print(f"         ❌ No generic names worked")

    except Exception as e:
        print(f"         ❌ Generic names strategy failed: {e}")

    return False, None, None


def check_tflite_model_safe(model_path):
    """
    Safe TFLite model checker that avoids private methods
    """
    print(f"🔍 Checking TFLite model: {os.path.basename(model_path)}")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False

    try:
        # # Load the TFLite model
        # interpreter = tf.lite.Interpreter(model_path=model_path)
        # # Try to load with Flex delegate
        # # import tflite_runtime.interpreter as tflite
        # # interpreter = tflite.Interpreter(model_path=model_path)
        # interpreter.allocate_tensors()
        # print("✅ Model loaded successfully!")


        # alternative
        # Load with TensorFlow Select ops support
        # try:
        #     # Method 1: Use TensorFlow's TFLite interpreter with select ops
        #     interpreter = tf.lite.Interpreter(
        #         model_path=model_path,
        #         experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_flex_delegate.so')]
        #     )
        # except:
        #     # Method 2: Alternative approach
        #     interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter = load_model_with_multiple_methods(model_path=model_path)#load_model_with_select_ops(model_path=model_path)

        interpreter.allocate_tensors()

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Get model information
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    signatures = interpreter.get_signature_list()

    print(f"\n📊 Model Structure:")
    print(f"   Signatures: {list(signatures.keys())}")
    print(f"   Total inputs: {len(input_details)}")
    print(f"   Total outputs: {len(output_details)}")

    # Detailed input analysis
    print(f"\n📥 Input Details:")
    for i, detail in enumerate(input_details):
        shape_str = str(detail['shape']).replace('-1', 'batch')
        print(f"   Input {i}: '{detail['name']}' - Shape: {shape_str} - Type: {detail['dtype']}")

    print(f"\n📤 Output Details:")
    for i, detail in enumerate(output_details):
        shape_str = str(detail['shape']).replace('-1', 'batch')
        print(f"   Output {i}: '{detail['name']}' - Shape: {shape_str} - Type: {detail['dtype']}")

    # Test different signature input strategies
    if signatures:
        print(f"\n🧪 Testing Signatures...")

        for sig_name in signatures.keys():
            print(f"\n   Testing signature: '{sig_name}'")

            success, method, working_inputs = test_signature_strategies(
                interpreter, sig_name, input_details
            )

            if success:
                print(f"   ✅ Success with method: {method}")
                return True, method, working_inputs, sig_name
            else:
                print(f"   ❌ All signature methods failed for '{sig_name}'")

    # Fallback: Manual tensor setting
    print(f"\n🔧 Testing Manual Tensor Setting...")

    try:
        manual_inputs = []
        for detail in input_details:
            actual_shape = [1 if dim == -1 else dim for dim in detail['shape']]

            # Create appropriate test data
            if 'snc' in detail['name'].lower() or len(actual_shape) >= 2:
                test_data = np.random.normal(0, 0.1, actual_shape).astype(detail['dtype'].type)
            else:
                test_data = np.random.random(actual_shape).astype(detail['dtype'].type)

            interpreter.set_tensor(detail['index'], test_data)
            manual_inputs.append((detail['name'], test_data))
            print(f"   Set input '{detail['name']}': shape {actual_shape}")

        # Run inference
        interpreter.invoke()

        # Get outputs
        results = {}
        for detail in output_details:
            output_data = interpreter.get_tensor(detail['index'])
            results[detail['name']] = output_data
            if hasattr(output_data, 'shape') and output_data.size > 1:
                print(f"   Output '{detail['name']}': shape {output_data.shape}, values {output_data.flatten()[:3]}...")
            else:
                print(f"   Output '{detail['name']}': {output_data}")

        print("   ✅ Manual tensor method successful!")
        return True, 'manual_tensors', manual_inputs, None

    except Exception as e:
        print(f"   ❌ Manual tensor method failed: {e}")
        return False, None, None, None


def load_model_with_multiple_methods(model_path):
    """Try multiple methods to load TFLite model with Select ops support"""

    print(f"🔄 Attempting to load model with Select ops support...")

    # Method 1: Try with 'select_tf_ops' delegate name
    try:
        print("   Method 1: Using 'select_tf_ops' delegate...")
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                tf.lite.experimental.load_delegate('select_tf_ops')
            ]
        )
        interpreter.allocate_tensors()
        print("   ✅ Success with 'select_tf_ops' delegate")
        return interpreter, "select_tf_ops"
    except Exception as e:
        print(f"   ❌ Method 1 failed: {str(e)[:100]}...")

    # Method 2: Try with flex delegate
    try:
        print("   Method 2: Using flex delegate...")
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                tf.lite.experimental.load_delegate('libtensorflowlite_flex_delegate.so')
            ]
        )
        interpreter.allocate_tensors()
        print("   ✅ Success with flex delegate")
        return interpreter, "flex_delegate"
    except Exception as e:
        print(f"   ❌ Method 2 failed: {str(e)[:100]}...")

    # Method 3: Try with TensorFlow Select ops delegate (alternative path)
    try:
        print("   Method 3: Using alternative select ops path...")
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                tf.lite.experimental.load_delegate('tensorflow_select_tf_ops')
            ]
        )
        interpreter.allocate_tensors()
        print("   ✅ Success with alternative select ops")
        return interpreter, "tensorflow_select_tf_ops"
    except Exception as e:
        print(f"   ❌ Method 3 failed: {str(e)[:100]}...")

    # Method 4: Try to create delegate with explicit library path
    try:
        print("   Method 4: Trying with explicit library paths...")

        # Common paths where the delegate might be located
        possible_paths = [
            'libtensorflowlite_select_tf_ops.so',
            'libtensorflowlite_flex_delegate.so',
            '/usr/local/lib/python*/site-packages/tensorflow/lite/delegates/select_tf_ops/libtensorflowlite_select_tf_ops.so',
            f'{os.path.dirname(tf.__file__)}/lite/delegates/select_tf_ops/libtensorflowlite_select_tf_ops.so'
        ]

        for delegate_path in possible_paths:
            try:
                print(f"      Trying path: {delegate_path}")
                interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[
                        tf.lite.experimental.load_delegate(delegate_path)
                    ]
                )
                interpreter.allocate_tensors()
                print(f"   ✅ Success with path: {delegate_path}")
                return interpreter, f"explicit_path:{delegate_path}"
            except:
                continue

        print("   ❌ Method 4 failed: No valid delegate paths found")
    except Exception as e:
        print(f"   ❌ Method 4 failed: {str(e)[:100]}...")

    # Method 5: Try with TensorFlow 2.x approach (using converter options)
    try:
        print("   Method 5: Using built-in TensorFlow with implicit Select ops...")

        # This might work if TensorFlow was built with Select ops support
        interpreter = tf.lite.Interpreter(model_path=model_path)

        # Try to allocate tensors - if SelfAdjointEigV2 is supported, this should work
        interpreter.allocate_tensors()
        print("   ✅ Success with built-in TensorFlow (Select ops implicitly supported)")
        return interpreter, "builtin_implicit"

    except Exception as e:
        print(f"   ❌ Method 5 failed: {str(e)[:100]}...")

        # Check if it's specifically the SelfAdjointEigV2 error
        if "SelfAdjointEigV2" in str(e):
            print("   🔍 Confirmed: SelfAdjointEigV2 operation not supported")

    print("   ❌ All methods failed to load model with Select ops support")
    return None, None
# Your specific model check
def check_your_weight_estimation_model(model_path):
    """
    Check your specific weight estimation model
    """

    print("🎯 Checking your weight estimation model...")
    print("=" * 50)

    success = check_tflite_model_safe(model_path)

    if success:
        print(f"\n🎉 SUCCESS! Your model is working!")
        print(f"\n📋 Here's the working code for your model:")

        # Generate working code template

        print(f"# Load your model")
        print(f"interpreter = tf.lite.Interpreter(model_path='{model_path}')")
        print("interpreter.allocate_tensors()")
        print("")
        print("# Get model details")
        print("input_details = interpreter.get_input_details()")
        print("output_details = interpreter.get_output_details()")
        print("signatures = interpreter.get_signature_list()")
        print("")
        print("# Method 1: Using signatures (if available)")
        print("if signatures:")
        print("    signature_name = list(signatures.keys())[0]  # Get first signature")
        print("    signature_runner = interpreter.get_signature_runner(signature_name)")
        print("    ")
        print("    # Create your input data (replace with real sensor data)")
        print("    inputs = {}")
        print("    for detail in input_details:")
        print("        input_name = detail['name']")
        print("        input_shape = [1 if dim == -1 else dim for dim in detail['shape']]")
        print("        inputs[input_name] = np.random.normal(0, 0.1, input_shape).astype(detail['dtype'].type)")
        print("    ")
        print("    # Run inference")
        print("    outputs = signature_runner(**inputs)")
        print("    print('Prediction:', outputs)")
        print("")
        print("# Method 2: Manual tensor setting (fallback)")
        print("else:")
        print("    # Set inputs")
        print("    for detail in input_details:")
        print("        input_shape = [1 if dim == -1 else dim for dim in detail['shape']]")
        print("        test_data = np.random.normal(0, 0.1, input_shape).astype(detail['dtype'].type)")
        print("        interpreter.set_tensor(detail['index'], test_data)")
        print("    ")
        print("    # Run inference")
        print("    interpreter.invoke()")
        print("    ")
        print("    # Get outputs")
        print("    for detail in output_details:")
        print("        output = interpreter.get_tensor(detail['index'])")
        print("        print(f'Output {detail[\"name\"]}: {output}')")
        print("```")

    else:
        print(f"\n❌ Issues found with your model.")
        print(f"Common solutions:")
        print(f"  1. Check if the model file exists and is not corrupted")
        print(f"  2. Make sure the model was converted properly to TFLite")
        print(f"  3. If you get 'SelfAdjointEigV2' errors, use the exact training + preprocessing approach")

    return success


def load_model_with_select_ops(model_path):
    """Load TFLite model with TensorFlow Select ops support"""
    try:
        # Load interpreter with select ops delegate
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                tf.lite.experimental.load_delegate('select_tf_ops')
            ]
        )
        interpreter.allocate_tensors()
        print("✅ Model loaded with TensorFlow Select ops support")
        return interpreter

    except Exception as e:
        print(f"❌ Failed to load with select ops: {e}")

        # Fallback: try loading with built-in TF support
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            print("✅ Model loaded with built-in TF support")
            return interpreter
        except Exception as e2:
            print(f"❌ Complete failure: {e2}")
            return None
# Load the TFLite model
# model_path="/home/wld-algo-6/Production/WeightEstimation2K/logs/04-06-2025-14-02-08/snc_weight_estimation_model.tflite"
# model_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/04-06-2025-15-52-35/snc_weight_estimation_model.tflite'
# model_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/05-06-2025-10-27-45/snc_weight_estimation_model.tflite'
model_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/05-06-2025-12-17-53/snc_weight_estimation_model.tflite'
model_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/08-06-2025-11-27-43/snc_weight_estimation_model.tflite'
model_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/12-06-2025-18-13-15/snc_weight_estimation_model.tflite'

# Run the check
check_tflite_model_safe(model_path)

# print('check_your_weight_estimation_model')
# check_your_weight_estimation_model(model_path)

