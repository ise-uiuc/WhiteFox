
class Model(torch.nn.Module):
    def forward(self, in1, in2, in3, in4, in5):
        t0 = torch.mm(in3, in1)
        t1 = torch.mm(in2, in4)
        out = t0 + t1
        out = torch.mm(in2.t(), in2)
        out = out + torch.mm(in1.t(), in1)
        return out
# Inputs to the model
in1 = torch.randn(5, 5)
in2 = torch.randn(5, 5)
in3 = torch.randn(5, 5)
in4 = torch.randn(5, 5)
in5 = torch.randn(5, 5)
