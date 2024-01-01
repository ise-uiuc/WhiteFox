
class Model(torch.nn.Module):
    def forward(self, A, B):
        t1 = torch.mm(A, B)
        t2 = torch.mm(A, A)
        t3 = T.relu(t1 + t2)
        t4 = torch.mm(t2, B)
        t5 = t2 + t3
        t6 = torch.mm(t4, A)
        t7 = t5 + t6
        return t7
# Inputs to the model
A = torch.rand(3, 3)
B = torch.rand(3, 3)
