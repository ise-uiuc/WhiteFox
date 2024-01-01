
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 3))
        self.bias = torch.nn.Parameter(torch.randn(3))
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.weight, self.bias)
        v1 = v0.permute(0, 2, 1)
        return v1
# Inputs to the model
x0 = torch.randn(1, 2, 2, 2)
