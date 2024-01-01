
class Model(torch.nn.Module):
    def forward(self, x1):
        t1 = torch.mm(x1, x1)
        t2 = torch.mm(t1, x1)
        t3 = torch.mm(t2, t2)
        t4 = torch.mm(t3, t2)
        t5 = torch.mm(t4, t4)
        t6 = torch.mm(t5, t2)
        t7 = t2 + t6
        t8 = torch.mm(t6, t4)
        return t7 + t8
# Inputs to the model
input = torch.randn(128, 128)
