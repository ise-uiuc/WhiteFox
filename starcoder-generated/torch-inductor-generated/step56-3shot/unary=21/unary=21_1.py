
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.conv1_weight = torch.nn.Parameter(torch.randn(64, 64, 1, 1))
        self.conv2_weight = torch.nn.Parameter(torch.randn(64, 64, 1, 1))
        self.conv3_weight = torch.nn.Parameter(torch.randn(1, 64, 1, 1))
    def forward(self, input):
        x = torch.conv2d(input, self.conv1_weight, stride=1, padding=0, dilation=1, groups=1)
        x = self.tanh(x)
        x = torch.conv2d(x, self.conv2_weight, stride=1, padding=0, dilation=1, groups=1)
        x = self.tanh(x)
        x = torch.conv2d(x, self.conv3_weight, stride=1, padding=0, dilation=1, groups=1)
        v1 = self.tanh(x)
        return v1
# Inputs to the model
input = torch.randn(1, 64, 1, 1)
