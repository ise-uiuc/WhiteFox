
class Model(torch.nn.Module):
    def forward(self, A, B, C, D, E, F):
        t1 = torch.mm(A, torch.mm(B, torch.mm(C, torch.mm(D, E))))
        t2 = torch.mm(F, torch.mm(E, torch.mm(D, torch.mm(C, B))))
        t2 = t2 + torch.mm(D, torch.mm(C, torch.mm(B, torch.mm(A, F))))
        return t1 + t2
# Inputs to the model
A = torch.randn(4, 4)
B = torch.randn(4, 4)
C = torch.randn(4, 4)
D = torch.randn(4, 4)
E = torch.randn(4, 4)
F = torch.randn(4, 4)
