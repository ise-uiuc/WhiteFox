
model = Model()

# Create the input tensor for the model
input_names = [ "input_0" ]
output_names = [ "output" ]
input_dims = [[1, 3, 16, 16]]
#input_dims = [[64, 64, 32, 32]]
input = torch.randn(input_dims[0])
# Input created

# Apply fusion optimization.
torch_mlir.graph_runner.optimize_model(model, input, input_names, output_names)
# Model after optimization
