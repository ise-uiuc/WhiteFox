
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        x3 = F.dropout(x1, p=0.5)
        x4 = F.dropout(x1, p=0.5)
        x5 = F.dropout(x1, p=0.5)
        x6 = torch.rand_like(x1)
        x7 = torch.rand_like(x1)
        return x5
# Inputs to the model
x1 = torch.randn(1)
