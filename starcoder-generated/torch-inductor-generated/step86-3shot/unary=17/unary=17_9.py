
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.sigmoid(x1)
        v2 = v1.mul(3)
        v3 = v2.transpose(3, 2)
        v4 = v3.transpose(3, 2)
        return v4.transpose(4, 3)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
