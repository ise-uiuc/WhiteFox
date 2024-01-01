
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x + torch.randn_like(x) + x + torch.randn_like(x)
        return y
# Inputs to the model
x = torch.randn(2, 2, 3, 4)
