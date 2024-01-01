
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(1, 1)
        )
    def forward(self, x1):
        x2 = self.submodule(x1)
        x3 = torch.nn.functional.dropout(x2, 0.4)
        x4 = torch.rand_like(x3)
        return x4
# Inputs to the model
x1 = torch.randn(10, 1)
