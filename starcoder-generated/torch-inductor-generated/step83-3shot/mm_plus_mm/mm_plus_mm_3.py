
class ModelMult(torch.nn.Module):
    def forward(self, A, B):
        t1 = torch.mm(A, B)
        t2 = torch.mm(A, B)
        t3 = torch.mm(A, B)
        t4 = torch.mm(A, B)
        return t1 + t2 + t1 + t2
# Inputs to the model
A = torch.randn(3, 5)
B = torch.randn(3, 5)
