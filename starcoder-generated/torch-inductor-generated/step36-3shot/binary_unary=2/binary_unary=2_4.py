
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 20, 4, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(20, 15, 5, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 9
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 25, 25)
