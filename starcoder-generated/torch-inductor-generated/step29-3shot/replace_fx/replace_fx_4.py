
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        p1 = torch.nn.functional.dropout(x1, p=0.2, training=True)
        q1 = torch.nn.functional.dropout(x2, p=0.1, training=True)
        r1 = torch.pow(p1, q1)
        s1 = torch.relu(r1)
        t1 = torch.rand_like(r1, dtype=torch.float)
        return r1, s1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
