
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        mm1 = torch.mm(input1, input2)
        mm2 = torch.mm(input3, input4)
        add_op = (mm1 + mm2) * (input2 * input4).mm(input2 * input4)
        res = torch.mm(add_op, input2 * input4)
        return add_op.mm(input2.mm(input4))
# Inputs to the model
input1 = torch.randn(55, 55)
input2 = torch.randn(55, 55)
input3 = torch.randn(55, 55)
input4 = torch.randn(55, 55)
