
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(5, 5, 2, stride=2)
        self.conv1 = torch.nn.ConvTranspose2d(5, 5, 5, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(x1)
        v3 = torch.relu(v1)
        v4 = torch.relu(v2)
        v5 = torch.sigmoid(v3)
        v6 = torch.sigmoid(v4)
        return v5, v6
# Inputs to the model
x1 = torch.randn(1, 5, 5, 5)
