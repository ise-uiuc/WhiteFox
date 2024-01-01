
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        z = x + y
# Inputs to the model
x = torch.randn(1, 3, 4)
