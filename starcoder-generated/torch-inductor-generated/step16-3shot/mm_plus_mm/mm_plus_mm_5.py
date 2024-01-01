
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input5, t2)
        return t1 + t2 + t3
# Inputs to the model
input1 = torch.randn(5, 159)
input2 = torch.randn(5, 113)
input3 = torch.randn(5, 104)
input4 = torch.randn(5, 14)
input5 = torch.randn(5, 14)
