
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(1, 4, kernel_size = 3, padding=1)
    def forward(self, x):
        v1 = self.conv1d(x)
        v2 = self.conv1d(x)
        v3 = self.conv1d(x)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x = torch.randn(1, 1, 28)
