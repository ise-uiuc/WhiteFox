
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        x = torch.rand_like(x3)
        x = F.dropout(x1, p=0.5)
        x = torch.rand_like(x2)
        return x1 + x2 + x3
# Inputs to the model
x1 = torch.randn((2, 1, 2))
x2 = torch.randn(1, 2, 3)
x3 = torch.randn(3, 1, 3)
