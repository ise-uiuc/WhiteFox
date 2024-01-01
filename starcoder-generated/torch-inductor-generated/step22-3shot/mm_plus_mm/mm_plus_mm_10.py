
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        mm1 = torch.mm(input3, input4)
        mm2 = torch.mm(input1, input2)
        t = mm1 + mm2
        return t.mm(input3.mm(input4))
# Inputs to the model
mm1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
input4 = torch.randn(8, 8)
