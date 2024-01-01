
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x + x
        x = x - 2 * x
        x = x * x
        x = x[0][0][0] + x[0][0][0]
        x = torch.cat([x, x, x], dim=0)
        x = x[0] + x[1] + x[2]
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
