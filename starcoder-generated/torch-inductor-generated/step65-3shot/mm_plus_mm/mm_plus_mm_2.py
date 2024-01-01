
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input5, input6)
        t4 = t1 + input1
        t5 = torch.mm(input1, input5)
        t6 = t3 + input3
        t7 = torch.mm(input2, input6)
        t8 = t2 - t5
        t9 = torch.mm(input3, input5)
        t10 = t7 - t9
        t11 = t4 + t6
        t12 = t8 + t10
        out = torch.mm(t11, t12)
        return out
# Inputs to the model
input1 = torch.randn(24, 24)
input2 = torch.randn(24, 24)
input3 = torch.randn(24, 24)
input4 = torch.randn(24, 24)
input5 = torch.randn(24, 24)
input6 = torch.randn(24, 24)
