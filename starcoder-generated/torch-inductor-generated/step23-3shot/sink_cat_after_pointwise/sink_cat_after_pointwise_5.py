
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(2, -1)
        x = torch.cat((x, x), dim=1)
        x = torch.cat((x, x), dim=0)
        return x
# Inputs to the model
x = torch.randn(5, 3, 4)
