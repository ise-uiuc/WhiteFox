
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        mm1 = torch.mm(input1, input2)
        mm2 = torch.mm(input3, input4)
        mm3 = torch.mm(input2, input4)
        mm4 = torch.mm(input5, input4)
        t = mm1 + mm2
        return t.mm(input2.mm(input4).mm(input5) + input2.mm(input4).mm(input3))
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
input5 = torch.randn(5, 5)
