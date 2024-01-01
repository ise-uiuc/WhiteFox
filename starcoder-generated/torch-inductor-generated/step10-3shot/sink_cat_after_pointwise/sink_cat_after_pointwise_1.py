
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = x.clone()
        for i in range(0, 1):
            if i == 1:
                x = x.repeat(3, 1, 1)
        for i in range(0, 1):
            x.tanh()
        for i in range(0, 1):
            if i == 1:
                x = torch.cat((z, x), dim=1)
        for i in range(0, 1):
            if i == 1:
                x = x.clone()
        y = x.abs()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
