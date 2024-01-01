
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t0 = torch.mm(input1, input2)
        t1 = torch.mm(input2, input1)
        t2 = torch.mm(input3, input2)
        t3 = torch.mm(input3, input1)
        t4 = t2 + t1
        t5 = t0 + t3
        t6 = torch.mm(input1, input3)
        return t6 + t5 + t4
# Inputs to the model
input1 = torch.randn(10, 8, 8)
input2 = torch.randn(10, 8, 8)
input3 = torch.randn(10, 8, 8)
input4 = torch.randn(10, 8, 8)
