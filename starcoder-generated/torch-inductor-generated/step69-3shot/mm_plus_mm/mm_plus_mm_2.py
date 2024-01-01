
class Model(torch.nn.Module):
    def forward(self, A):
        t1 = torch.mm(A, A)
        t2 = torch.mm(A, A)
        return torch.mm(t1, t2)
# Inputs to the model
A = torch.rand(3, 3)
