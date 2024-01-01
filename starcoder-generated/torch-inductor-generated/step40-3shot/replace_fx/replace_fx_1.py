
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return torch.nn.functional.dropout(torch.rand_like(x1), p=0.1)
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = torch.rand_like(x1)
        x2 = x1 + 6
        return F.dropout(x2, p=0.2, training=True)
class Model3(torch.nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
    def forward(self, x1):
        x2 = self.model1(x1)
        x3 = self.model2(x2)
        return x3

m1 = torch.nn.Sequential(
        Model1(),
        Model2()
    )
m2 = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        Model3(m1, m1)
    )
p1 = torch.randn(2, 2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
