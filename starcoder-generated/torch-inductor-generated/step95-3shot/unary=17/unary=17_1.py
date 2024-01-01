
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=(3, 32), stride=(2, 1), padding=(1, 0))
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 64), stride=(1, 1), padding=(1, 0))
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
