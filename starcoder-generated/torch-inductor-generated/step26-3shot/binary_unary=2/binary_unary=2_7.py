
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.utils.weight_norm(torch.nn.Conv2d(3, 32, 3), name="weight")
        self.conv2 = torch.nn.Conv2d(32, 2, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(2, 1, 12, stride=1, padding=0)
    def reset_parameters(self):
        torch.nn.init.zeros_(self.conv1.weight_g)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 - 0.1
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 1.5
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        return v7
# Inputs to the model
x = torch.randn(64, 3, 100, 100)
