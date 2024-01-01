
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 11
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 22)
