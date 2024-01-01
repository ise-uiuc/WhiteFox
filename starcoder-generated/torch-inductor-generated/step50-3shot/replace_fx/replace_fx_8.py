
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.nn.functional.dropout(x, p=0.5)
        b = a.sum()
        return b
# Inputs to the model
x1 = torch.randn(32, 32)
