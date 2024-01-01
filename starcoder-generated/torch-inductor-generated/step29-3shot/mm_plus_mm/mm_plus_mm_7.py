
class Model(torch.nn.Module):
    def forward(self, a1, input1, input2, input3, input4):
        t1 = torch.mm(a1, input4)
        t2 = torch.mm(input4, input2)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
a1 = torch.randn(16, 16)
