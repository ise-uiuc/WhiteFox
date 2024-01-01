
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(48, 48)
    def forward(self, x):
        y = self.linear(x)
        if y.shape == (y.shape[0], 5):
            y = y.relu()
        return y
# Inputs to the model
x = torch.randn(4, 2, 48)
