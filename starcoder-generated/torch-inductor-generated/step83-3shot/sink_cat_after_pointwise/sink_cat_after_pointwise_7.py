
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.size(0), -1)
        x = torch.cat((y, y, x), dim=1)
        x = x.tanh()
        x = x.sigmoid()
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
