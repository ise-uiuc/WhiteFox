
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, t2)
        t2 = torch.mm(input, t2)
        t3 = torch.mm(input, t2 + t3)
        t4 = torch.mm(t1, t2 * t3)
        t5 = torch.mm(t3, t4)
        t6 = t3
        t7 = torch.mm(t3, t5)
        t8 = torch.mm(t1, t2 + t3 * t4)
        return t1 + t2 + t3
# Inputs to the model
input = torch.randn(64, 64)
