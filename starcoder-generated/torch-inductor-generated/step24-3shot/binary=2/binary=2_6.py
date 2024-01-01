
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 14, kernel_size=1, stride=2, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(14, 100, kernel_size=2, stride=2, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(100, 44, kernel_size=2, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = F.relu(v1)
        v3 = v2[:, :, :-1, :-1]
        v4 = self.conv2(v2)
        v5 = F.relu(v4)
        v6 = self.conv3(v5)
        v7 = self.conv3(v2)
        v8 = v6 - v7
        return v8
# Inputs of the model
x = torch.randn(1, 3, 32, 32)
