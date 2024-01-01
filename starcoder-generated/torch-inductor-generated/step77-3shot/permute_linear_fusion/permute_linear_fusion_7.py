
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(1, 2, 3, 4, 0)
        v1 = torch.flatten(v1, 1)
        v1 = v1.permute(0, 2, 1)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3, 3, 3, 3)
