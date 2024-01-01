
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, (1,), groups=1)  
    def forward(self, x1):
        v3 = torch.nn.functional.conv1d(x1, weight=self.conv.weight, bias=self.conv.bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
# Inputs to the model
x1 = torch.randn(1, 2, 3, 2)
