
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        torch.nn.init.zeros_(self.linear.weight)
    def forward(self, x):
        x1 = self.linear(x)
        x2 = torch.rand_like(x)
        x3 = F.dropout(x2, 0.5)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
