
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = x.tanh()
        x = x.view(x.shape[0], -1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
