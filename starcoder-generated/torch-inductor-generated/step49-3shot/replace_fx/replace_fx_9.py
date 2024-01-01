
class Model(torch.nn.Module):
    def __init__(self, a: float):
        super().__init__()
        self.a = a

    def forward(self, x):
        x = torch.nn.functional.dropout(x, p=self.a)
        return torch.rand_like(x)
# Inputs to the model
x1 = torch.randn(2, 3)
