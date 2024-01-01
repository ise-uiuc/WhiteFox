
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v0 = x1.permute(0, 2, 1)
        v1 = torch.bmm(v0, x1)
        return v1.shape
# Inputs to the model
x1 = torch.randn(1, 4, 4)
