
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose2 = torch.nn.ConvTranspose2d(1024, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x1):
        v1 = self.convtranspose2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1024, 4, 4)
