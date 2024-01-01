
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convtranspose = torch.nn.ConvTranspose2d(1, 1, 27, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.convtranspose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 124, 124)
