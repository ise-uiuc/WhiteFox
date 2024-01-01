
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.sig = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.sigmoid()
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
