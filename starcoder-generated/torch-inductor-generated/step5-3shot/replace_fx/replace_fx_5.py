
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, x):
        x1 = x.mean(dim=0, keepdim=True)
        x2 = x - x1
        x3 = x2.sum()
        x4 = torch.add(x3, x2)
        x5 = torch.sub(x3, x2)
        return x
# Inputs to the model
x1 = torch.randn(1, 1)
