
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 11, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(3, 11, 1, stride=1, padding=1)
    def forward(self, input, padding0=None, padding1=None):
        x1 = self.conv0(input)
        if padding0 == None:
            padding0 = torch.randn(x1.shape)
        x2 = self.conv1(input)
        if padding1 == None:
            padding1 = torch.randn(x2.shape)
        x3 = x1 + x2
        return x3
# Inputs to the model
input = torch.randn(1, 3, 64, 64)
