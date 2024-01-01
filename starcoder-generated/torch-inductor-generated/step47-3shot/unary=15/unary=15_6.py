
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, (7, 20), stride=(1, 5), padding=(3, 10))
        self.conv2 = torch.nn.Conv2d(16, 8, (20, 4), stride=(5, 1), padding=(1, 2))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 1024, 8)
