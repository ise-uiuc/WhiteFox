
class Model(nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6, input7, input8):
        m1 = torch.mm(input2, input3)
        m2 = torch.mm(input5, input6)
        m3 = torch.mm(input7, input8)
        t1 = torch.mm(m1, m3)
        t2 = torch.mm(m1, m2)
        return torch.mm(t1, input4) + torch.mm(t2, input1)
# Inputs to the model
input1 = torch.randn(2, 2)
m1 = torch.randn(2, 2)
m2 = torch.randn(2, 2)
m3 = torch.randn(2, 2)
input4 = torch.randn(2, 2)
input5 = torch.randn(2, 2)
input6 = torch.randn(2, 2)
input7 = torch.randn(2, 2)
input8 = torch.randn(2, 2)
