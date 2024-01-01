
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 8, 2, stride=1)
        self.conv2 = torch.nn.Conv1d(8, 8, 3, stride=2, padding=1, groups=8)
        self.conv3 = torch.nn.Conv1d(8, 16, 1)
        self.conv4 = torch.nn.Conv1d(16, 16, 2, stride=1, groups=8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
# Inputs to the model
x = torch.randn(1, 3, 64)
