
class Model(torch.nn.Module):
    def forward(self, a1, a2, a3, a4, a5, a6, a7, a8):
        t1 = torch.mm(a4, a2)
        t3 = t1 + a8.permute(1, 0)
        t2 = torch.mm(a3, a4)
        t4 = t2 + a7
        t5 = t3 + t4
        return t5
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
input5 = torch.randn(5, 5)
input6 = torch.randn(5, 5)
input7 = torch.randn(5, 5)
input8 = torch.randn(5, 5)
