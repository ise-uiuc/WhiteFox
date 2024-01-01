
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        for i in range(0, 2):
            y.tanh()
            if i == 1:
                y = y.view(2, -1)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
