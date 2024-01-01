
class Model(torch.nn.Module):
    def forward(self, in1, in2, in3, in4, in5, in6):
        t0 = torch.mm(in1, in2) + torch.mm(in3, torch.mm(in4, in5))
        return t0 + in6
# Inputs to the model
in1 = torch.randn(4, 4)
in2 = torch.randn(4, 4)
in3 = torch.randn(4, 4)
in4 = torch.randn(4, 4)
in5 = torch.randn(4, 4)
in6 = torch.randn(4, 4)
