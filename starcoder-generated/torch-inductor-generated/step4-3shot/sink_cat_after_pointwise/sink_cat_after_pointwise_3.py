
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        z = x.cat()
        x = y + z
        return x
# Inputs to the model
x = torch.randn(1)
y = torch.randn(1)
