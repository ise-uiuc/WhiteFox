
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v0 = x1.permute(0, 2, 1)
        return v0
# Inputs to the model
x1 = torch.randn(1, 2, 2)
