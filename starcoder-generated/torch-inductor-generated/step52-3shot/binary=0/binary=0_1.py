
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d((8,8), (4,4))
    def forward(self, x1):
        v1 = self.pool(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
