
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 6, 1, stride=1, padding=1)
    def forward(self, input=None, padding1=None, other=None,  t3 = None):
        if input is None:
            input = torch.randn(1, 9, 64, 64)
        x1 = self.conv(input)
        if other == None:
            other = torch.randn(x1.shape)
        v2 = x1 + other
        if padding1 == None:
            padding1 = torch.randn(v2.shape)
        v3 = v2 + padding1
        if t3 == None:
            t3 = torch.randn(v3.shape)
        v4 = v3 + t3
        return v4
# Inputs to the model
input = torch.randn(1, 9, 64, 64)
