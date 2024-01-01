
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 5, stride=1, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1[:, :, 2:, :]
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 20, 30)
