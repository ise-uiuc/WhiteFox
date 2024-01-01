
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.randn(1, 2, 2)
        x2 = F.dropout(x)
        t = torch.rand_like(x)
        return t.view(1, 1).shape
# Inputs to the model
x = torch.randn(8, 3)
