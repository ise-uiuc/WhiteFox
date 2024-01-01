
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = v1[:, :, :2]
        return v2[:, 0, :]
# Inputs to the model
x1 = torch.randn(1, 2, 3)
