
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        mm1 = torch.mm(input2, input1.t())
        mm2 = torch.mm(input3, input4.t())
        t = mm1 + mm2
        return t.mm(input3.mm(input1))
# Inputs to the model
mm1 = torch.randn(8, 8)
mm2 = torch.randn(8, 8)
mm3 = torch.randn(8, 8)
mm4 = torch.randn(8, 8)
