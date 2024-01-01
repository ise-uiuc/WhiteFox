
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose2d = torch.nn.ConvTranspose2d(3, 4, kernel_size=(1, 4), stride=(3, 2), padding=(0, 1))
    def forward(self, x1):
        v1 = self.convtranspose2d(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 100, 200)
