
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v3 = torch.ones(1)
        v4 = torch.Tensor.flatten(v3)
        v1 = torch.nn.functional.linear(v4.view(2, 2), self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cpu')
