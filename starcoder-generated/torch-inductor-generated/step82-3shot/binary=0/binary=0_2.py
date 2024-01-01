
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=1, padding2=1, x3=1):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.rand(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.rand(1, 3, 128, 128)
