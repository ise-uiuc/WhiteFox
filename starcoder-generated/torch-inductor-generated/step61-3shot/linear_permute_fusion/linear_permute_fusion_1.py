
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = torch.nn.AdaptiveAvgPool2d(output_size=(1, 2))
        self.flatten = torch.nn.Flatten()
        self.conv = torch.nn.Conv1d(1, 3, 2, padding=0, stride=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, requires_grad=True)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(3, 3, bias=True)
    def forward(self, x3):
        v3 = self.pooling(x3)
        v4 = self.flatten(v3).squeeze(1)
        v5 = self.conv(v4)
        v6 = self.relu(v5)
        v7 = self.linear(v6)
        return v7.unsqueeze(1)
# Inputs to the model
x3 = torch.randn(1, 1, 3, 3)
