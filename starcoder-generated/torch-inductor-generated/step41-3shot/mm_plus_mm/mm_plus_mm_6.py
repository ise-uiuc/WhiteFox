
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input2)
        t3 = torch.mm(input1, input2) + t1
        t4 = torch.mm(input1, input2) + t2
        t5 = torch.mm(input1, input2) + t1
        t6 = torch.mm(input1, input2) + t1 + t3
        t7 = torch.mm(input1, input2) - torch.mm(input1, input2) - torch.mm(input1, input2) + torch.mm(input1, input2)
        t8 = torch.mm(input1, input2) - torch.mm(input1, input2) - torch.mm(input1, input2) + torch.mm(input1, input2)
        t9 = torch.mm(input1, input2) - t7 - t6
        return torch.mm(input1, input2) - torch.mm(input1, input2) - torch.mm(input1, input2) + torch.mm(input1, input2) - torch.mm(input1, input2) + torch.mm(input1, input2)
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
