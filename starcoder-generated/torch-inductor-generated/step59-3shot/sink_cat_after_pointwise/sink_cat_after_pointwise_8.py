
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x, x), dim=1)
        return y[:, 0:2]
# Inputs to the model
x = torch.randn(1, 4)
