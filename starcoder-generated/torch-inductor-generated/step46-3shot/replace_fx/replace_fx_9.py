
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1, dropout_p=0.1)
        x3 = F.interpolate(x1, x2, mode='nearest')
        x4 = F.dropout(x1, p=0.5)
        x5 = F.interpolate(x3, x4, mode='nearest')
        return x4
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x3 = F.interpolate(x1, scale_factor=x1.size()[-1], mode='nearest')
        x2 = torch.rand_like(x3)
        x4 = F.interpolate(x1, x2, mode='nearest')
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
