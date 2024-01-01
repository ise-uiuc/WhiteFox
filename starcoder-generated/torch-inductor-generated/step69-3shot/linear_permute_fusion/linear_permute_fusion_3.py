
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 10)
        self.linear1 = torch.nn.Linear(10, 2) # Add linear layer for another linear transformation.
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias) # Apply linear transformation to the input tensor first.
        v2 = v1.permute(0, 2, 1) # Apply another linear transformation to the input tensor next.
        v3 = torch.nn.functional.linear(v2, self.linear1.weight, self.linear1.bias) # Apply another linear transformation to the input tensor.
        return v3
# Inputs to the model
x1 = torch.randn(10, 2, 2)
