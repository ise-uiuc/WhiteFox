
class Model(torch.nn.Module):
    def forward(self, A, B):
        t1 = torch.mm(A, B)
        t2 = torch.mm(A, A)
        t2 = t2 + torch.mm(B, B)
        t3 = torch.mm(t1, t1)
        return t1 + t2 + t3
# Inputs to the model
A = torch.randn(3, 3)
B = torch.randn(3, 3)
