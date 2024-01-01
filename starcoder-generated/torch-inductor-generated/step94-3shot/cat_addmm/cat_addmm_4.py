
# Torch script model definition
@torch.jit.script
def forward(x: torch.Tensor) -> torch.Tensor:
    # Note that the input has to be flattened in order to successfully apply mm.
    x = x.view(x.shape[0], -1)
    # Apply first linear layer weights and inputs to perform matmul operation.
    x = matmul(weights, x, out=x)
    # Pass the inputs through another linear layer to get the resulting output.
    x = matmul(weights2, x, out=x)
    # Return the final output after applying tanh activation function.
    return torch.tanh(x)
# Inputs to the model
x = torch.randn(2, 2)
