
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input2, input3)
        t3 = torch.mm(input4, t1)
        t4 = torch.mm(input1, t2)
        t5 = torch.mm(input1, t4)
        t6 = torch.mm(t3, t3)
        out = input5 + t6 + input6
        return out
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
input5 = torch.randn(16, 16)
input6 = torch.randn(16, 16)
