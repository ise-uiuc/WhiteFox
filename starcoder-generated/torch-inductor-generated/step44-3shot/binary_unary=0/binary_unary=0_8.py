
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.tanh(x1)
        v2 = x1 + 0
        return v1
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
