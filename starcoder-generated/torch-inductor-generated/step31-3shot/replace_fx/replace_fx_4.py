, inputs are arbitrary
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, t1):
        t0 = torch.nn.functional.dropout(t1)
        t2 = torch.rand_like(t0)
        t3 = torch.nn.functional.dropout(t2)
        return t3
