
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        w1 = torch.rand(1, 1, 1)
        y = x * w1
        w2 = torch.rand(1, 1, 2, 3)
        x_bchw = x + y + w2
        return x_bchw
# Inputs to the model
x = torch.randn(1, 1, 2, 3)
