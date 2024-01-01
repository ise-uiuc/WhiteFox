
class Model(torch.nn.Module):
    def forward(self, A, B, C, D):
        E = A + B    # 1
        F = C + D    # 2
        H = torch.mm(E, F) + torch.mm(E, F) # 3
        return H
# Inputs to the model
A = torch.randn(3, 5)
B = torch.randn(5, 3)
C = torch.randn(3, 5)
D = torch.randn(5, 3)
