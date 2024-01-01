
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y=None):
        c1 = torch.nn.functional.dropout(x, p=0.2)
        c2 = torch.nn.functional.dropout(x, p=0)
        a = torch.pow(6, c1) * c2
        return a
# Inputs to the model
x1 = torch.randn(1, 5, 10, 10)
x1_t = torch.randn(1, 5)
x = torch.randn(1)
