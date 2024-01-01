
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        mm1 = torch.mm(input1, input2)
        mm2 = torch.mm(input3, input4)
        t = mm1 + mm2
        return torch.mm(t, torch.mm(input2, input4))
# Inputs to the model
mm1 = torch.randn(55, 55)
input2 = torch.randn(55, 55)
input3 = torch.randn(55, 55)
input4 = torch.randn(55, 55)
