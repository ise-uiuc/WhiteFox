
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input4) + torch.mm(input4, input5)
        t2 = torch.mm(input2, input1) + torch.mm(input3, input2)
        t3 = torch.mm(t1, t2)
        t4 = t1 + t2
        t5 = t4 + t3
        return torch.mm(t5, t4) * t4
# Inputs to the model
input1 = torch.randn(3, 2)
input2 = torch.randn(3, 2)
input3 = torch.randn(2, 6)
input4 = torch.randn(2, 2)
input5 = torch.randn(2, 3)
