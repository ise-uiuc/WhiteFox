
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(3, 16, (6, 6), stride=(4, 4))
    def forward(self, x1):
        v1 = self.convt(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
