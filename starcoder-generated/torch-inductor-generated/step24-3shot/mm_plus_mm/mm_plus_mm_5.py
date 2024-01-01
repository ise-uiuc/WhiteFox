
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input2, input3)
        t3 = torch.tanh(t1) + t2 + torch.mm(input1, input2)
        t4 =  t1 + 2.0 * t2 + 3.0 * torch.mm(input3, input4)
        t5 = t3 * t4
        return t5.sum()
# Inputs to the model
input1 = torch.randn(2, 5)
input2 = torch.randn(2, 5)
input3 = torch.randn(5, 2)
