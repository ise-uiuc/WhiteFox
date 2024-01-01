
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input3)
        # t2 = torch.mm(input1, input1)
        # t3 = torch.mm(input2, input4)
        # t4 = torch.mm(input3, input4)
        t5 = torch.mm(input1, input4)
        t6 = torch.mm(input2, input5)
        t7 = torch.mm(input3, input5)
        return t1 + t2 + t3 + t4 + t5 + t6 + t7
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
input4 = torch.randn(8, 8)
input5 = torch.randn(8, 8)
