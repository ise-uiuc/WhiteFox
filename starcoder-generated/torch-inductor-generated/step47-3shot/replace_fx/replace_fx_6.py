
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x):
        a1 = self.linear(x)
        a2 = self.linear(a1)
        a3 = torch.flatten(a2, start_dim=0, end_dim=1)
        return torch.nn.functional.dropout(a3, p=0.5)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
