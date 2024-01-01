
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        self.linear = torch.nn.functional.gelu(x)
        return self.linear
# Inputs to the model
x1 = torch.randn(3, 3)
