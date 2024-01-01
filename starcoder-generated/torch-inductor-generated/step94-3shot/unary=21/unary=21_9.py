
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(31, 57, 3, stride=2, padding=1, dilation=1, groups=1)
        self.conv2 = torch.nn.Conv1d(57, 3, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        return v4.expand_as(x)
# Inputs to the model
tensor = torch.randn(10, 31, 100)
