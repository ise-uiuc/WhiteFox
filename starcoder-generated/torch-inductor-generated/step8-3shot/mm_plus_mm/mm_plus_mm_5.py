
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input2, input1)
        t2 = torch.mm(input4, input1)
        t3 = torch.mm(input2, input3)
        t4 = torch.mm(input4, input3)
        t5 = t1 + t2
        t6 = t3 + t4
        return t5 * t6
# Inputs to the model
input1 = torch.randn(2, 3)
input2 = torch.randn(3, 5)
input3 = torch.randn(2, 5)
input4 = torch.randn(3, 5)
