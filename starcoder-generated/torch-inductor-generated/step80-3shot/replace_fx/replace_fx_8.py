
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = F.dropout(x1, p=0.5)
        x2 = F.dropout(x2, p=0.5)
        x = torch.rand_like(x1)
        x += x2
        return x
# Inputs to the model
x1 = torch.randn(1, 1, 2)
x2 = torch.randn(1, 1, 2)
