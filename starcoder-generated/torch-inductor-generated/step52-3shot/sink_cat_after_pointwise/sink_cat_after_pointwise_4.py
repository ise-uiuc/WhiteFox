
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.size(0), -1)
        x = torch.cat((y, y), dim=1)
        return torch.tanh(x)
# Inputs to the model
x = torch.randn(2, 2, 2)
