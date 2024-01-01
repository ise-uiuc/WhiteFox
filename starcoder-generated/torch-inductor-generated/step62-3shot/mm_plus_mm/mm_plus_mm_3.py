
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        s1 = t1 - t2
        t3 = torch.mm(s1, input5)
        return t3
# Inputs to the model
input1 = torch.randn(14, 14)
input2 = torch.randn(7, 14)
input3 = torch.randn(14, 14)
input4 = torch.randn(37, 14)
input5 = torch.randn(7, 7)
