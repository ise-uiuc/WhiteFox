
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = F.conv2d(x1, weight=self.conv.weight, bias=self.conv.bias, stride=1, padding=self.conv.padding, dilation=self.conv.dilation, groups=(self.conv.in_channels), padding_mode=(self.conv.padding_mode))
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
