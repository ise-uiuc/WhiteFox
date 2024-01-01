
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d((66,56))
    def forward(self, x1, other=True):
        v1 = self.pool(x1)
        if other == True:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 48, 48)
