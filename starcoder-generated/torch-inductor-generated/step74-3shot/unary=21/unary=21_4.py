
class ModelTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, (3, 3), stride=1, padding=1, dilation=1, groups=1, bias=False) 
    def forward(self, x):
        t1 = self.conv(x)
        t2 = nn.Tanh()(t1) 
        return t2
# Inputs to the model
x = torch.randn(1, 1, 10, 10)
