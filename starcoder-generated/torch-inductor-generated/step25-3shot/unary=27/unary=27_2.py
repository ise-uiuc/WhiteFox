
class Model(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 8, kernel_size=1)
        self.relu = torch.nn.ReLU()
        self.min = min
    def forward(self, input, min, max):
        v1 = self.conv(input)
        v2 = self.relu(v1)
        v3 = torch.clamp_min(v2, min)
        v4 = torch.clamp_max(v3, max)
        return v4
min = 50
# Inputs to the model
input = torch.randn(1, 3, 64, 64)
min = 4.5
max = 2.0
