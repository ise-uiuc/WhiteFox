
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2 * x1
        return v1
# Inputs to the model
x1 = torch.randn(1)
x2 = torch.randn(1)
