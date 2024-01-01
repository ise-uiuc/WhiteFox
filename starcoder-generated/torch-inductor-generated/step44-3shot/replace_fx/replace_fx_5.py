
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x):
        a = self.linear(x)
        return torch.nn.functional.dropout(a, p=0.5) + torch.nn.functional.dropout(a, p=0.5) + torch.nn.functional.dropout(a, p=0.5)
# Inputs to the model
x1 = torch.randn(1, 2, 4)
