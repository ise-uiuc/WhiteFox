
class Model(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=torch.rand(1)*(1.05**self.a))
        x3 = torch.rand_like(x1)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
