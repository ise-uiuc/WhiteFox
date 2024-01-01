
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input2, input3)
        t2 = torch.mm(input1, input2)
        return t1 ^ t2
# Inputs to the model
input1 = torch.randn(3, 5)
input2 = torch.randn(3, 2)
input3 = torch.randn(2, 1)
input4 = torch.randn(1, 3)
