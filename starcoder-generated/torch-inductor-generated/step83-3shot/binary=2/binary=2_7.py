
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v2 = x - 0.64
        return v2
# Inputs to the model
x = torch.randn(1, 3, 16, 16)
