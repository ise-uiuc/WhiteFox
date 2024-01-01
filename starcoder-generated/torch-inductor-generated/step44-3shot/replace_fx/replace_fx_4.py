
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1):
        a = self.linear(x1)
        return F.dropout(a, p=0.5, training=False)
# Inputs to the model
x1 = torch.randn(1, 3)
