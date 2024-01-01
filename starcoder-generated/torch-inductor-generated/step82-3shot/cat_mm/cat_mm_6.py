
class Model(torch.nn.Module):
    def __init__(self, dim1, dim2, dff):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim1, dff),
            torch.nn.Tanh(),
            torch.nn.Linear(dff, dff),
            torch.nn.Tanh(),
            torch.nn.Linear(dff, dim2),
        )
    def forward(self, x1, x2):
        return torch.cat([self.mlp(x1), self.mlp(x2)], -1)
# Inputs to the model
x1 = torch.randn(2, 16, 2)
x2 = torch.randn(2, 16, 2)
