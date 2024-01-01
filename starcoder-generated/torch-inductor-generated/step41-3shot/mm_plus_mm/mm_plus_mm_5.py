
class Model(torch.nn.Module):
    def forward(self, in1, in2, in3, in4):
        t0 = torch.mm(in1, in2)
        t1 = torch.mm(in3, in4)
        t2 = torch.mm(in4, in5)
        t0 += t1
        t2 += t1
        t0 += torch.mm(in5, in6)
        t2 += t2
        return t0 + t2
# Inputs to the model
in1 = torch.randn(6, 6)
in2 = torch.randn(6, 6)
in3 = torch.randn(6, 6)
in4 = torch.randn(6, 6)
