
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        x = y.view(y.shape[0], -1)
        x = x.tanh() # Removed'relu' since only ReLU is unary operations
        return x
# Inputs to the model
x = torch.randn(2, 1, 2, 4)
