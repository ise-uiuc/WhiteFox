
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.zeros(x1.size(0), 14)
        x3 = torch.cat([x1, x2], 1)
        x = torch.cat([x1,x3], 1)
        return x
# Inputs to the model
x1 = torch.randn(2, 6)
