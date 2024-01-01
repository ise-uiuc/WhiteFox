
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 16, (3, 7), stride=(2, 4), padding=(1, 2), groups=2)
        self.conv2 = torch.nn.Conv2d(8, 16, (3, 7), stride=(2, 4), padding=(1, 2), groups=8)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.relu(v1)
        v4 = torch.relu(v2)
        return torch.add(v3, v4)
# Inputs to the model
x1 = torch.randn(1, 8, 33, 125)
