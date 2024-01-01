
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input2, input1)
        t3 = torch.mm(input1, input1)
        t4 = torch.mm(1. + input2, input2 + 1. + input2)
        t5 = torch.mm(t1, t2)
        return t1 + t2, t3 + t4, t5
# Inputs to the model
input1 = torch.randn(5, 3)
input2 = torch.randn(3, 5)
