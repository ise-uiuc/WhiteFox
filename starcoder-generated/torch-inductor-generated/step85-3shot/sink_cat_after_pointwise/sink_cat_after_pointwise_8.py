
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        for i in range(4):
            f = torch.sin(y)
            y = y + f.sin()
            y = y * f.tanh()
            y = y - f.permute(2, 0, 1).sin().flatten().sin().view(1, 2, 3).cos()

        return y
# Inputs to the model
x = torch.randn(1, 2, 3)
