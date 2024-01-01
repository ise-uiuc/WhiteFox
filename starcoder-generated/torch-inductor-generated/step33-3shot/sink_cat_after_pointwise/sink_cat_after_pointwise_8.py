
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.tanh(x)
        x = torch.cat((x, x), dim=1)
        return torch.add(x, 1)
# Inputs to the model
x = torch.randn(2, 2, 2)
