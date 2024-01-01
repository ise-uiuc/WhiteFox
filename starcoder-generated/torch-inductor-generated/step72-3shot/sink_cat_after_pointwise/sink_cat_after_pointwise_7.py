
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = x.view(-1)
        y = torch.cat((z, z), dim=0)
        return y.tan()
# Inputs to the model
x = torch.randn(2, 4, 3, 2)
