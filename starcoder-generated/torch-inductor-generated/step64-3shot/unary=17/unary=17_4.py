
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 32, (3, 3), padding=(1, 1), stride=(2, 2))
        self.conv1 = torch.nn.ConvTranspose2d(32, 3, (3, 3), padding=(1, 1), stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
