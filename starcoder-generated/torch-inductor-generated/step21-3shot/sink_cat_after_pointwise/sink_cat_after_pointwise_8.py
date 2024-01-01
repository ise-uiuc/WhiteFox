
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.relu(x)
        x = y.view(x.shape[0], -1)
        x = torch.tanh(x)
        x = x.view(x.shape[0], -1)
        x = x.tanh()
        return x
# Inputs to the model
x = torch.randn(3, 2, 4)
