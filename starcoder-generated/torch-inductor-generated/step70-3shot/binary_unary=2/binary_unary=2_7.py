
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 5, stride=4, padding=1)
    def forward(self, x2):
        v1 = self.conv1(x2)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = self.conv2(v3.unsqueeze(0))
        v5 = v4 - 0.5
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x2 = torch.randn(1, 3, 16, 16)
