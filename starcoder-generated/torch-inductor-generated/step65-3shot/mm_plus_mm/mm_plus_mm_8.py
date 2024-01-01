
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input1, input4)
        t3 = t1 + t2

        t4 = torch.mm(input2, input3)
        t5 = t1 + t4

        t6 = torch.mm(input2, input4)
        t7 = torch.mm(input3, input4)
        t8 = t6 + t7

        out = t3 + t5 + t8
        return out
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
