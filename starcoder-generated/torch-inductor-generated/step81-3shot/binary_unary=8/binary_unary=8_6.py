
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(5, 5, 3, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv1(x)
        v3 = self.conv1(x)
        v5 = v1 + v2 + v3
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 5, 32)
