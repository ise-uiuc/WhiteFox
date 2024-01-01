
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6, input7):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input2, input4)
        t2 = torch.mm(input2, input4)
        return t1 + t2 + input5 + input5 + input6 + input7
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
input4 = torch.randn(3, 3)
input5 = torch.randn(5,)
input6 = torch.randn(5,)
input7 = torch.randn(5,)
