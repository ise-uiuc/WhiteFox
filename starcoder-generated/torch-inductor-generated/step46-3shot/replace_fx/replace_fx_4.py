
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.nn.functional.dropout(x1, p=0.5)
        return x2
class Model(torch.nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        self.linear = torch.nn.Parameter(torch.randn(num_class, x2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(5)
