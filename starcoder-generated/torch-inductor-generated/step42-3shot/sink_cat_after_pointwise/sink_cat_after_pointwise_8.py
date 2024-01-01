
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1).view(-1).relu() if (x.shape[1] * x.shape[1])!= 0 else torch.cat((x, x), dim=1).view(-1).tanh()
        return y
# Inputs to the model
x = torch.randn(3, 0)
