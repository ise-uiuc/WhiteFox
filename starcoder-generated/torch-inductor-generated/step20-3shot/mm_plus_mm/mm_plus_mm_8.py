
class Model(nn.Module):
    def forward(self, input1, input2):
        a = torch.rand(10,10)
        b = torch.rand(10,10)
        c = torch.rand(10,10)
        return a.mm(b).mm(a.mm(c)).mm(c).mm(torch.rand(10, 10)).mm(a)*0.01
# Inputs to the model
input1 = torch.randn(10, 10)
input2 = torch.randn(10, 10)
