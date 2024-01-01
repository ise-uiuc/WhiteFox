
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y = y.view(y.shape[0], -1)
        if y.shape == (2, 12):
            y = y.tanh()
        elif y.shape == (2, 8):
            y = torch.relu(y)
        return torch.tanh(y) if y.shape == (2, 4) else torch.sin(y)
# Inputs to the model
x = torch.randn(2, 3, 4)
