
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t = t1 + t2
        t3 = torch.sigmoid(t)
        t4 = torch.mm(input4, input5)
        t5 = torch.sigmoid(t4)
        t6 = torch.mm(input5, input3)
        t7 = torch.sigmoid(t6)
        t8 = torch.mm(input5, input2)
        t9 = torch.sigmoid(t8)
        output = torch.mm(input5, (t3 + t4 + t5 + t6 + t7))
        return t9 + output
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
input4 = torch.randn(3, 3)
input5 = torch.randn(3, 3)
