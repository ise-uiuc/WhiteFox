
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1[0:x.size()[0],0:x.size()[1],:])
        return v2
# Inputs to the model
x = torch.randn(10, 10, 28, 28)
