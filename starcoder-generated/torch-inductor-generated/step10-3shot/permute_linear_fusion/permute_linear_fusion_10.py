
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(1, 0, 2)
        v2 = v1.permute(0, 2, 1)
        v3 = v2.permute(1, 2, 0)
        return v3
# Inputs to the model
x1 = torch.randn(2, 1, 1)
