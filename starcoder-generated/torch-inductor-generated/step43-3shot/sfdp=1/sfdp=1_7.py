
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(100, 100)
 
    def forward(self, qb, kb, vb):
        mat = torch.matmul(qb, kb.transpose(-2, -1))
        f1 = self.m1(mat)
        f = f1.div(0.001)
        d1 = torch.nn.functional.dropout(f, p=0.2)
        output = d1.matmul(vb)
        return output

# Initializing the model
m = Model()

# Initializing the input tensors
qb = torch.randn(100, 100)
kb = torch.randn(800, 100)
vb = torch.randn(800, 100)
