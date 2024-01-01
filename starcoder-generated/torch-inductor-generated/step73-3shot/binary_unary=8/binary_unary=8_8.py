
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2= torch.nn.Conv2d(32, 2, 3, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = torch.relu(v1 + v2 + x)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
