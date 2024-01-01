
class Model(torch.nn.Module):
    def forward(self, A, B, C, D):
        out = torch.mm(A, B) + torch.mm(C, D)
        a = torch.mm(out, B)
        b = torch.mm(out, D)
        out = torch.mm(A, B) + torch.mm(C, D)
        return a + b + out
# Inputs to the model
A = torch.rand(3, 2)
B = torch.rand(2, 3)
C = torch.rand(4, 5)
D = torch.rand(5, 5)
