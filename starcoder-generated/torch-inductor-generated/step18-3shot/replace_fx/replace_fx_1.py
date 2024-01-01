
class MyModel(torch.nn.Module):
    def __init__(self, dim=32):
        super(MyModel, self).__init__()
        self.proj = torch.nn.Linear(30, dim)
    def forward(self, z):
        t1 = torch.rand_like(z)
        t2 = torch.nn.functional.dropout(t1, p=0.2)
        t3 = torch.rand_like(z)
        return t3 + t2 + self.proj(t3)
# Inputs to the model
z1 = torch.zeros([1, 30])
