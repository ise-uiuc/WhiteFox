
class Model(nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.matmul(input1, input2)
        t2 = torch.matmul(input3, input4)
        t3 = t1 + t2
        t4 = torch.mm(input1, input4) + input1
        return t3 + t4
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 4)
