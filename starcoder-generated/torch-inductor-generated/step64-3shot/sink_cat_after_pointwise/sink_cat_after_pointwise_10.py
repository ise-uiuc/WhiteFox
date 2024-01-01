
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat((x, x, x), dim=1).view(3, -1).relu() if x.shape[0]!=3 else torch.cat((x, x, x), dim=1).view(3, -1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
