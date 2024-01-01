
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.sum(x)
        a1 = torch.nn.functional.dropout(x, p=0.3)
        a2 = torch.nn.functional.dropout(a1, p=0.05)
        return a
# Inputs to the model
x = torch.randn(1)
