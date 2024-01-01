
class Model(torch.nn.Module):
    def forward(self, A, B, C, D):
        t1 = torch.mm(A, B)
        t2 = torch.mm(C, D)
        t3 = t1 + t2
        return t3
# Inputs to the model
A = torch.rand(3, 3)
B = torch.rand(3, 3)
C = torch.rand(3, 3)
D = torch.rand(3, 3)
