
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = torch.sigmoid(input1)
        t3 = torch.mm(t1, t2)
        t4 = torch.mm(input1, input3)
        t5 = torch.mm(input2, input3)
        t6 = torch.mm(input3, input1)
        t7 = t6 + t1 + t5 + t4
        return t3
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 2)
input3 = torch.randn(2, 2)
