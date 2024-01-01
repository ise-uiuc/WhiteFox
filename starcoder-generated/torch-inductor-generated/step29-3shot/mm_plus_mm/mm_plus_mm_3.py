
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1.transpose(-1, 1), input3.transpose(-1, 1))
        t2 = torch.mm(input1.transpose(-1, 1), t1)
        t3 = torch.mm(t2, input4)
        return t2 + t3
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
input4 = torch.randn(8, 8)
