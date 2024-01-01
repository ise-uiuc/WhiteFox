
class Model(nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input3, input4)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 4)
