
class m1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.mm(x1, x1)
        x3 = torch.randint(0, 9, (1,))
        x4 = torch.rand_like(x2)
        return torch.nn.functional.dropout(x2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
