
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x):
        a = self.linear(x)
        return torch.cat(([a]*4), dim=1)
# Inputs to the model
x1 = torch.randn(1, 4, 2)
