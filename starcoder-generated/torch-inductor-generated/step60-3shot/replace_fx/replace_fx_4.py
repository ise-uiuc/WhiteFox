
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
    def forward(self, x1):
        x1 = self.linear(x1)
        t1 = F.dropout(x1, p=0.5, training=True)
        t2 = torch.rand_like(t1, dtype=torch.float)
        return t2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
