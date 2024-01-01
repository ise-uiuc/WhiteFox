
class Model(nn.Module):
    def forward(self, input1, input2):
        v1 = torch.mm(input1, input1)
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input1)
        t3 = torch.mm(input1, input2)
        t4 = torch.mm(input1, input2)
        t5 = torch.mm(input1, input2)
        v2 = torch.mm(input1, input1)
        return torch.mm(input1, input1) + t5 + v2
# Inputs to the model
input1 = torch.randn(10, 10)
input2 = torch.randn(10, 10)
