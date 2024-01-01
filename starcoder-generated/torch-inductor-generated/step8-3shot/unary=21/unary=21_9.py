
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1)

        self.conv.weight=self.conv.weight/torch.max(self.conv.weight)
    def forward(self, x):
        y = self.conv(x)
        t = torch.tanh(y)
        return t
# Inputs to the model
x = torch.randn(2, 3, 1024, 100)
