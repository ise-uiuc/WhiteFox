
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 7, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 7, 1, stride=1, padding=0)
    def forward(self, x1, conv1=True, x2=None):
        v1 = self.conv1(x1)
        if conv1 == True:
            other = self.conv2(x2)
        if x2 == None:
            x2 = torch.randn(v1.shape)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
x2 = torch.randn(1, 16, 16, 16)
