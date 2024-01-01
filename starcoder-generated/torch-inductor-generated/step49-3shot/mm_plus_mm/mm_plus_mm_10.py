
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        mm1 = torch.mm(input1, input2)
        mm2 = torch.mm(input2, input1)
        t2 = mm2 + input1
        t3 = mm1 + t2
        t5 = t2 + t3
        t6 = t3 + t2
        t7 = t6 + input2
        return t7
# Inputs to the model
input1 = torch.randn(80, 97)
input2 = torch.randn(97, 80)
