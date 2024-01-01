
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input2, input3)
        t3 = torch.mm(input1, input2)
        t4 = torch.mm(input1, input3)
        return torch.mm(input1, input2) + torch.mm(input2, input3) + t1 + t2 + t3 + t4
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 2)
input3 = torch.randn(2, 2)
