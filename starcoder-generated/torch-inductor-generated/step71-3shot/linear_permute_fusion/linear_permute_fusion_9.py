
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1[:, :, 0]
        v2 = v1.mean()
        return v1
# Inputs to the model
x1 = torch.randn(2, 2, 4)
