
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 1, stride=1)
    def forward(self, x1, other=None, padding1=None, weight=None):
        v1 = self.conv(x1)
        if weight == None:
            weight = torch.randn(v1.shape)
        if other == None:
            other = torch.randn(v1.shape)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = torch.add(other, padding1, v2)
        return torch.add(v3, weight)  
# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
