
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose2d(7, 9, 2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.convtranspose(x1)
        v3 = v1 + torch.sigmoid(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 7, 224, 224)
