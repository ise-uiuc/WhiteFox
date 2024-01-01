
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = x.permute(0, 2, 1)
        a2 = torch.flip(x, [1])
        return torch.nn.functional.dropout2d(x, 0.91)
# Inputs to the model
x = torch.randn(4, 7, 8)
