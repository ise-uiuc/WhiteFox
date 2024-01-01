
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 32, (5, 5), stride=(1, 1), padding=(2, 2))
        self.conv1 = torch.nn.ConvTranspose2d(32, 1, (4, 4), stride=(2, 2), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
