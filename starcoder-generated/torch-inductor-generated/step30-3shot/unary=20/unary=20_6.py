
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose2 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=2, kernel_size=(260, 2), stride=260)
    def forward(self, x1):
        v1 = self.convtranspose2(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 600, 1)
