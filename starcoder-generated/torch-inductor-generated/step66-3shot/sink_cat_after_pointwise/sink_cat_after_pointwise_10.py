
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.cat([x, x, x, x, x], dim=0)
        shape1 = list(x1.shape)
        shape1[0] = -1
        x2 = x1.view(shape1[1], -1)
        x = x2[:3]
        return x.sin()
# Inputs to the model
x = torch.randn(2, 3, 4)
