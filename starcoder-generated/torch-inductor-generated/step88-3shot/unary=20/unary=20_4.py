
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU(256)
    def forward(self, x1):
        v1 = self.prelu(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 256, 50, 50)
