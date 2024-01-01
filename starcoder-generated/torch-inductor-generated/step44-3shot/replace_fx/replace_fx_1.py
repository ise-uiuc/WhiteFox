
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x):
        a1 = torch.nn.functional.dropout(self.linear(x))
        return torch.rand_like(a1)
# Inputs to the model
x1 = torch.randn(1, 2, 4)
