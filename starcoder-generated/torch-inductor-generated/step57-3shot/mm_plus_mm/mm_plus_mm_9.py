
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, a1, a2):
        t1 = torch.mm(a1, input2)
        t2 = torch.mm(input1, a2)
        t3 = torch.mm(input4, a1)
        t4 = torch.mm(input4, a1)
        return t1 + t2 + t3 + t4
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 4)
a1 = torch.randn(4, 4)
a2 = torch.randn(4, 4)
