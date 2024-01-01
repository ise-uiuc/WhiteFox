
class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = x[0:-1]
        x2 = x[1:]
        x3 = torch.nn.functional.gelu(x1)
        return x2
# Inputs to the model
x1 = torch.randn(1)
