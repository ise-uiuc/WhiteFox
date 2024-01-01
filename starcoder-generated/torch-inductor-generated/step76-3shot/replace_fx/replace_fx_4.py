
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x2 = F.dropout(x, p=0.5)
        x1 = torch.rand_like(x)
        t1 = F.dropout(x, p=0.5)
        return torch.arange((x2 + x1 + t1).size())
# Inputs to the model
x = torch.randn(1, 2, 2)
