
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, other=torch.tensor(1.0)):
        v1 = torch.nn.functional.linear(x1, other)  # Apply a linear transformation to the input tensor
        v2 = v1 + other # Add another tensor (specified as an argument) to the output of the linear transformation
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1,5)
