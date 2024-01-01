
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 2, 4, stride=1, padding=0, bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(2, 4, 4, stride=2, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
