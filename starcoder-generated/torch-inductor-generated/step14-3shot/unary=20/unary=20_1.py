
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose2d(1, 1, (3, 3), stride=(1, 1), bias=False)
    def forward(self  x):
        v0 = F.pad(x, (1, 1, 1, 1), 'constant', 0)
        v1 = self.convtranspose(v0)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
