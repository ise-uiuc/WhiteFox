
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=10, out_features=3)
    def forward(self, x):
        t1 = torch.rand_like(x, layout=torch.strided)
        t2 = t1.view(t1.size(0), t1.size(1), 1, 1)
        t3 = t2 + t1.view(t1.size(0), t1.size(1), 1, 1)
        t4 = torch.nn.functional.dropout(t1)
        t5 = t3 + t4
        x1 = self.lin(t5)
        x2 = torch.rand_like(x1)
        x3 = torch.nn.functional.dropout(x1, p=0.7)
        return x3.flatten()
# Inputs to the model
x = torch.rand((1, 10))
