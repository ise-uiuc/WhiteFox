
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x):
        a1 = torch.nn.functional.dropout(x, p=0.3)
        a2 = torch.nn.functional.dropout(x, p=0.05)
        a3 = self.linear(a1)
        return a3
# Inputs to the model
x = torch.randn(10,10)
