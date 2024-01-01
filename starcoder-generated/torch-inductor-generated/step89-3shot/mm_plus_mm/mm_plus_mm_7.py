
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input3)
        return torch.mm(t1, input2.mm(input4[:,:7].mm(input2)))
# Inputs to the model
input1 = torch.randn(3, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 6)
