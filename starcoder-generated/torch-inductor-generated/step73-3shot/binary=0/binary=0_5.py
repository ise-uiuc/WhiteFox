
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 10, 3, stride=1, padding=1)
    def forward(self, x1, other=3, other2=None, other3=None, other4=None):
        v1 = self.conv1(x1)
        if other2 == None:
            other2 = torch.randn(v1.shape)
        v2 = v1 + other2
        if other3 == None:
            other3 = torch.randn(v1.shape)
        v3 = v2 + other3
        if other4 == None:
            other4 = torch.randn(v1.shape)
        v4 = v3 + other4
        return v4
# Inputs to the model
x1 = torch.randn(3, 7, 28, 28)
