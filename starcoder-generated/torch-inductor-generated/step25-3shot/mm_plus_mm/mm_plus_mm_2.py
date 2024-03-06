
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input1, input2)
        t3 = torch.mm(input2, input3)
        t4 = torch.mm(input1, input3)
        t5 = torch.mm(input2, input3)
        t6 = t1 + t2 + t3 + t4 + t5
        return t6
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)