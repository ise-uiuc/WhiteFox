
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        return t1.transpose(1, 2) + t2.T
# Inputs to the model
input1 = torch.randn(1, 1, 50, 50)
input2 = torch.randn(1, 1, 50, 50)
input3 = torch.randn(1, 1, 50, 50)
input4 = torch.randn(1, 1, 50, 50)
