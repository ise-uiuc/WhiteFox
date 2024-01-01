
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(10, 10, 16)
        self.conv2 = torch.nn.Conv1d(10, 10, 16)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 + v2
        return v3
# Inputs to the model
x = torch.randn(1, 10, 16)
