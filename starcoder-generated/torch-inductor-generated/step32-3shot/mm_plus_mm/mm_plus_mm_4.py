
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(torch.mm(input1, input1), input2) + torch.mm(input1, input2)
        t3 = torch.mm(input2, torch.mm(input1, input2)) + torch.mm(torch.mm(input1, input2), torch.mm(input1, input1))
        t4 = torch.mm(torch.mm(input2, torch.mm(input1, input2)), torch.mm(input1, input1)) + torch.mm(torch.mm(input2, torch.mm(input1, input1)), input2)
        t5 = torch.mm(torch.mm(torch.mm(input2, torch.mm(input1, input2)), torch.mm(input1, input1)), torch.mm(input1, input1)) + torch.mm(torch.mm(torch.mm(input2, torch.mm(input1, input1)), torch.mm(input1, input2)), torch.mm(input1, input1))
        t6 = t1 - (t2 + (t3 * t4) / t5)
        return t6
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
input4 = torch.randn(7, 7)
