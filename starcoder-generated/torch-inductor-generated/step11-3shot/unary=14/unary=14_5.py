
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.ReLU()
        self.convtranspose = torch.nn.ConvTranspose2d(14, 25, (7, 1), stride=(1, 1), padding=(0, 1))
    def forward(self, x1):
        x2 = self.a(x1)
        v1 = self.convtranspose(x2)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 14, 15, 47)
