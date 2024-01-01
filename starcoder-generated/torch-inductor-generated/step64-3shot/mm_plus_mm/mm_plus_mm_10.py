
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input3, input4) * torch.mm(input4, input2)
        t2 = torch.mm(input3, input4) * torch.mm(input4, input1)
        return t1 - t2 + torch.mm(input1, input2) - torch.mm(input1, input4) + torch.mm(input3, input2)
# Inputs to the model
input1 = torch.randn(1, 1)
input2 = torch.randn(1, 1)
input3 = torch.randn(1, 1)
input4 = torch.randn(1, 1)
