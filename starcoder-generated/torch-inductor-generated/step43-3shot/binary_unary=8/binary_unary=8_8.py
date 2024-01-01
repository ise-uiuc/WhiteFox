
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 64, kernel_size=(1,5), stride=1, padding=(0,2))
    def forward(self, v):
        v1 = torch.relu(self.conv(v))
        v2 = self.conv(v)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
v = torch.randn(1, 32, 28, 28)
