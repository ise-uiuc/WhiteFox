
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x * (x * x) + (x + x)
        return x
# Inputs to the model
x = torch.randn(20, 20)
