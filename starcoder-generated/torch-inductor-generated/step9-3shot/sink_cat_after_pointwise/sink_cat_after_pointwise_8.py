
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.relu(x)
        y = y.view(x.shape[0], -1)
        y = y.tanh()
        y = torch.cat((y, y), dim=1)
        x = y.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
