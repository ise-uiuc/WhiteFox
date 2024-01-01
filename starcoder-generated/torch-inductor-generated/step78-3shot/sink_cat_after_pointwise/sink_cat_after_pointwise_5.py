
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.tanh()
        x = y.cat((y, y), dim=1)
        return x
# Input to the model
x = torch.randn(3)
