
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x):
        y = self.linear1(x)
        y = torch.rand_like(y)
        t1 = torch.nn.functional.dropout(y, p=0.8)
        return t1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
