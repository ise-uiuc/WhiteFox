
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        z = y.view(y.shape[0], -1)
        x = z.tanh() if y.shape!= (2, 12) else z.relu()
        a, b, c = x.shape
        for _ in range(a):
            y = y + x
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
