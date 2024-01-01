
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v4 = x1
        v6 = v4.dtype
        v3 = torch.Size(torch.zeros([0], dtype=v6).shape)
        v2 = torch.zeros(v3, dtype=v6)
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 2, device='cpu')
