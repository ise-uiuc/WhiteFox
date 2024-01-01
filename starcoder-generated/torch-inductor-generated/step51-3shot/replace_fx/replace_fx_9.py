
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
    def forward(self, x):
        x = F.dropout(x, p=0.5)
        x = self.weight * x
        x = torch.rand_like(x)
        x = F.dropout(x, p=0.3)
        return x
# Inputs to the model
x1 = torch.randn(1,2, 2)
