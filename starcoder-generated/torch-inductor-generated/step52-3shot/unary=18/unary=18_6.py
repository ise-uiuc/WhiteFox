
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v2 = torch.sigmoid(x1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
