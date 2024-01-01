
t1 = linear(input_tensor) # Apply a linear transformation to the input tensor
t2 = torch.tanh(t1) # Apply the hyperbolic tangent function to the output of the linear transformation

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
