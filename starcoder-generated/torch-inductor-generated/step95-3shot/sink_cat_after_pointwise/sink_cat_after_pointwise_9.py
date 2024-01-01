
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x), dim=1)
        y = y.view(y.shape[0], y.shape[1], 3)
        y = y.relu()
        if y.shape!= (1, 3):
            y = y.tanh()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
