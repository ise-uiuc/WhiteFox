
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_3 = torch.nn.ConvTranspose3d(4, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.trans_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2, 2)
