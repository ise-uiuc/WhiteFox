
class Model(torch.nn.Module):
    def forward(self, A):
        t1 = torch.mm(A, A)
        t2 = torch.mm(t1, t1)
        t3 = t2 + t2
        return t1 + t2 + t3
# Inputs to the model
A = torch.randn(100, 100)
