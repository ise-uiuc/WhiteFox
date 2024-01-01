
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.Dropout2d(0.1)(x1)
        x3 = torch.rand_like(x1)
        return _common.ReduceSum((0, 1))(x3)
# Inputs to the model
x1 = torch.randn(1)
