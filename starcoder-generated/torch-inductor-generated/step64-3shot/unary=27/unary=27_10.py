
class Model(torch.nn.Module):
    def __init__(self, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 5, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(5, 1, 1, stride=1, padding=0)
        self.max = max
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv2(v1)
        v3 = torch.clamp_max(v2, self.max)
        return v3
max = 2.5
# Inputs to the model
x = torch.randn(1, 2, 10)
