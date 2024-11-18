from training_playground.custom.layers import SEMGScatteringTransform

# Create the transform
semg_transform = SEMGScatteringTransform()
# Build the filters (need to call build first)
semg_transform.build(input_shape=512)  # Example input shape
# Visualize the filter bank
semg_transform.plot_filter_bank()