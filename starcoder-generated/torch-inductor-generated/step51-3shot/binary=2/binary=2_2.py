
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2):
        v2 = x2 - 10.0
        return v2
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
