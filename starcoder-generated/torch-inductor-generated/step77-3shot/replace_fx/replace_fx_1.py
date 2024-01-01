
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
    def forward(self, x, x1, x2):
        x3 = torch.rand_like(x)
        t1 = torch.rand_like(x1)
        t2 = self.linear(torch.rand_like(x2))
        return torch.nn.functional.dropout(t1, p=0.2, training=True) + torch.nn.functional.dropout(t2, p=0.1, training=False) * 2.0 + torch.mean(x3, dim=[0])
# Inputs to the model
x = torch.randn(1, 3)
x1 = torch.randn(2, 1)
x2 = torch.randn(1, 1)
