
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input3, input3)
        t2 = torch.mm(input1, input1)
        t3 = torch.mm(input1, input1)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(545, 545)
input2 = torch.randn(545, 545)
input3 = torch.randn(545, 545)
input4 = torch.randn(545, 545)
