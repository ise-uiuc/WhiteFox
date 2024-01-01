
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1 + 2.0
        v4 = self.linear(v1)
        v2 = v4.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cpu')
