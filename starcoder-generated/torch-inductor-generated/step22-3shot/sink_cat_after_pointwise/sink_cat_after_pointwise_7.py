
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.relu(x)
        x = y.tanh() if x.shape == (2, 3, 4) else y.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
