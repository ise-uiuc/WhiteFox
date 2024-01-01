
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6, input7, input8):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = t1 + t2
        return t3
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
input5 = torch.randn(16, 16)
input6 = torch.randn(16, 16)
input7 = torch.randn(16, 16)
input8 = torch.randn(16, 16)
