
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d((1, 1), stride=1)
 
    def forward(self, x1):
        o1 = self.avgpool(x1)
        v4 = torch.nonzero(torch.isfinite(o1))
        return []

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
