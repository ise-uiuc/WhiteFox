
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3, bias=False)
        self.linear2 = torch.nn.Linear(3, 1, bias=False)
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(x1, scale_factor=1.0, mode='linear', align_corners=False)
        v3 = v1.permute(0, 2, 1)
        v4 = torch.matmul(v3, torch.nn.functional.gelu(self.linear.weight))
        v5 = torch.nn.functional.relu(v4)
        v6 = torch.nn.functional.linear(v5, self.linear2.weight)
        v7 = torch.tanh(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
