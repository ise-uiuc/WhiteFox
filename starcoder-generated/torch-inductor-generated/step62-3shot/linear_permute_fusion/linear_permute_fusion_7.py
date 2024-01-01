
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v2 = torch.nn.functional.linear(x1, torch.randn(2, 1), torch.randn(1))
        v4 = x1
        v1 = torch.nn.functional.linear(v4, v2, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return torch.nn.functional.linear(v2, v2, v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
