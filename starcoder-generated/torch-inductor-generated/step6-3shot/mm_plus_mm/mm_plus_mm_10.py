
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2) # Matrix multiplication between input1 and input2
        t2 = torch.mm(input3, input4) # Matrix multiplication between input3 and input4
        return t2 + t1
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
input4 = torch.randn(3, 3)
