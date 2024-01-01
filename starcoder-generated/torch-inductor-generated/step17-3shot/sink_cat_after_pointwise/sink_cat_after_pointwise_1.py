
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.relu()
        y = x.sigmoid() * y
        return y
# Inputs to the model
x = torch.randn(1, 2, 3)
