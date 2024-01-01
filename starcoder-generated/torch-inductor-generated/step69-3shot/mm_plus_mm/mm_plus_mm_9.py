
class ModelAdd(torch.nn.Module):
    def forward(self, A, B, C, D, E):
        t1 = torch.mm(A, B)
        t2 = torch.mm(C, D)
        t3 = t1 + t2
        t4 = torch.mm(t3, E)
        return t4
# Inputs to the model
A = torch.rand(3, 3)
B = torch.rand(3, 3)
C = torch.rand(3, 3)
D = torch.rand(3, 3)
E = torch.rand(3, 3)

