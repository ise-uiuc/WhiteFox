
class TransposeLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return torch.linear(x)
# Inputs to the model
x1 = torch.randn(2, 2, 2)
