
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        y = x.view(-1)
        return y
# Inputs to the model
x = torch.randn(2, 2, 2)
