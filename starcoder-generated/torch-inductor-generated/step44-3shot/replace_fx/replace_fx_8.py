
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = x * (F.dropout(x, p=0.5) + F.dropout(x, p=0.5) + F.dropout(x, p=0.5))
        return a1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
