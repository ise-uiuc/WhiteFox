
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        mm1 = torch.mm(input1, input2)
        mm2 = torch.mm(input3, input2)
        mm3 = torch.mm(input1, input4)
        mm4 = torch.mm(input3, input4)
        t1 = mm1 + mm2
        t2 = mm3 + mm4
        return t1 * t2
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
input4 = torch.randn(8, 8)
