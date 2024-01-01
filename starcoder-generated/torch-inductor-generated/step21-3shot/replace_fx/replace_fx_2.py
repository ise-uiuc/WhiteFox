
class model(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 16)
        self.linear2 = torch.nn.Linear(16, 4)
        self.p1 = p1
    def forward(self, x1):
        x2 = torch.nn.functional.gelu(self.linear1(x1))
        x3 = x2**self.p1
        x4 = torch.nn.functional.dropout(x3)
        x5 = self.linear2(x4)
        return x5
p1 = 1
# Inputs to the model
x1 = torch.randn(1, 4)
