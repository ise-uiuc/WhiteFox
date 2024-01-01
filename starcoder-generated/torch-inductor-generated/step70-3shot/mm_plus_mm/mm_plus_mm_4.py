
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        mm1 = torch.mm(input1, input2)
        mm2 = torch.mm(input2, input2)
        mm3 = torch.mm(input2, input3)
        mm4 = torch.mm(input3, input4)
        return mm1 + mm2 + mm3 + mm4
# Inputs to the model
input1 = torch.randn(6, 6)
input2 = torch.randn(6, 6)
input3 = torch.randn(6, 6)
input4 = torch.randn(6, 6)
