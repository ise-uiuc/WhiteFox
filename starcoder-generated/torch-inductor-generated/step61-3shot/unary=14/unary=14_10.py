
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = torch.nn.Conv1d(1, 1, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1d_1(x)
        return v1
# Inputs to the model
x = torch.randn(1, 1, 32)
