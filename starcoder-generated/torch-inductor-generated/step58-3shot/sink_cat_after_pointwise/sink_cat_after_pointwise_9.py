
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.cat([x, x, x, x], dim=0)
        shape1 = list(x.shape)
        shape1[0] = -1
        x2 = x1.view(-1, *x.shape[1:])
        x3 = x2.tanh()
        return x3
# Inputs to the model
x = torch.randn(2, 3, 4)
