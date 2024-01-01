
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0
        v3 = torch.nn.functional.relu(v2)
        v4 = v3.permute(0, 2, 3, 1)
        v5 = torch.flatten(v4, start_dim=1)
        return v5
# Inputs to the model
x1 = torch.randn(5, 1, 64, 64)
