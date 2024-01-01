
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        t1 = x1 + x5 + x6 + x7
        t2 = x2 + x3 + x4
        t3 = torch.mm(t1, t1)
        x = torch.mm(t2, t3)
        return x
# Inputs to the model
input1 = torch.randn(50,50)
input2 = torch.randn(50, 50)
input3 = torch.randn(50, 50)
input4 = torch.randn(50, 50)
input5 = torch.randn(50, 50)
input6 = torch.randn(50, 50)
input7 = torch.randn(50, 50)
