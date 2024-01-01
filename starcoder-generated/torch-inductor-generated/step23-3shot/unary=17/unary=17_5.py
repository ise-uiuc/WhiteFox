
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, (2,8), stride=(3,5), padding=(1,2))
        self.conv1 = torch.nn.ConvTranspose2d(16, 4, 3, padding=2, stride=3)
        self.conv2 = torch.nn.ConvTranspose2d(4, 1, 2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv2(v5)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
