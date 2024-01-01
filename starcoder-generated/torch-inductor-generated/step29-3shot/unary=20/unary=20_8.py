
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convTranspose0 = torch.nn.ConvTranspose2d(3, 64, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False, dilation=1, padding=(1, 1))
        self.convTranspose1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False, dilation=1, padding=(1, 1))
        self.convTranspose2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False, dilation=1, padding=(1, 1))
    def forward(self, x1):
        v1 = self.convTranspose0(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.convTranspose1(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.convTranspose2(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 480, 640)
