
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        x = y.view(-1)
        return x * 2
# Inputs to the model
x = torch.randn(1, 6, 3, 10)
