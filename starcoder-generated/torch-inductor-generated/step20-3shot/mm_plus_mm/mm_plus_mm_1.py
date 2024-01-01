
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2, input3, input4):
        mm1 = input1.mm(input2) + torch.mm(input3, input4)
        mm2 = input1.mm(input2) + input3.mm(input4)
        add_op = (mm1 + mm2).mm(input2.mm(input4))
        return add_op.mm(input2.mm(input4))
# Inputs to the model
mm1 = torch.randn(55, 55)
input2 = torch.randn(55, 55)
input3 = torch.randn(55, 55)
input4 = torch.randn(55, 55)
