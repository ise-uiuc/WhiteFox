
class Model(torch.nn.Module):
    def forward(self, A, B, C, D, E, F):
        res = torch.mm(A, B) + torch.mm(torch.mm(C, D), E)
        res = res + torch.mm(E, F) + torch.mm(A, F)
        return res
# Inputs to the model
A = torch.randn(4, 4)
B = torch.randn(4, 4)
C = torch.randn(4, 4)
D = torch.randn(4, 4)
E = torch.randn(4, 4)
F = torch.randn(4, 4)
