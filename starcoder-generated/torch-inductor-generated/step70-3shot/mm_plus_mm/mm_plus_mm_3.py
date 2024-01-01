
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input3)
        t3 = torch.mm(input1, input1)
        t3 = torch.mm(input5, input4)
        return torch.mm(t1, t2)
# Inputs to the model
input1 = torch.randn(7, 7)
input2 = torch.randn(7, 7)
input3 = torch.randn(7, 7)
input4 = torch.randn(7, 7)
