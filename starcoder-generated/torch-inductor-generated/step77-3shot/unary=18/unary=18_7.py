
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 13, (1, 1), stride=(1, 1))
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(13, 26, (1, 7), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 320)
