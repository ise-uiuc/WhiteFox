
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = torch.add(torch.sub(self.conv1(x1), 10), torch.relu(self.conv2(x1)))
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
