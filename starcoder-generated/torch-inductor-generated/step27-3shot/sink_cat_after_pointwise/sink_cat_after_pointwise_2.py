
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x * 2
        x = torch.cat((y, y), dim=1)
        return x.view(x.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(5, 3, 4)
