
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=int(1)) # padding is 1
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
