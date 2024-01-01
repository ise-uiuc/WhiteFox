
class ModelMul(torch.nn.Module):
    def forward(self, A, B, C, D, E):
        t1 = torch.mm(A, B)
        t2 = torch.mm(C, D)
        return torch.mm(t1, E) + torch.mm(t2, E)
# Inputs to the model
A = torch.rand(3, 5)
B = torch.rand(5, 5)
C = torch.rand(3, 5)
D = torch.rand(5, 5)
E = torch.rand(5, 2)
