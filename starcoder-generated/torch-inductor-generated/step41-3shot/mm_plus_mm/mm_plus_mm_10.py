
class Model(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4):
        t1 = torch.mm(in0, in1) + torch.mm(in2, torch.mm(in3, in4))
        return t1
# Inputs to the model
in0 = torch.randn(3, 3)
in1 = torch.randn(3, 3)
in2 = torch.randn(2, 3)
in3 = torch.randn(2, 2)
in4 = torch.randn(3, 2)
