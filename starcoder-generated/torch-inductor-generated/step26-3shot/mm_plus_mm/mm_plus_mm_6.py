
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input2, input4)
        t3 = torch.mm(input1, input4)
        t4 = torch.mm(input3, input2)
        t5 = t1 + t2
        t6 = t3 + t4
        return t5 * t6
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
