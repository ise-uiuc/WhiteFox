
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = x1.unsqueeze(0)
        x3 = x2 + x2.narrow(0, 1, 1)
        x4 = x1 + x2
        return x2, x3, x4
# Inputs to the model
x1 = torch.randn(2, 2)
