
class Model(torch.nn.Module):
    def forward(input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input1)
        t3 = torch.mm(input3, input4)
        t4 = torch.mm(input1, input4) + torch.mm(input2, input1) + torch.mm(input2, input3) + torch.mm(input3, input4) + torch.mm(input3, input1) + t2
        return torch.mm(t4, t1)
# Inputs to the model
input1 = torch.randn(1, 1)
input2 = torch.randn(1, 1)
input3 = torch.randn(1, 1)
input4 = torch.randn(1, 1)


