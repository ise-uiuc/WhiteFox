
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t = input1.mm(input2)
        t1 = t.mm(input3)
        t2 = t.mm(input4)
        t3 = torch.mm(t1 + t2, input2.mm(input4))
        return t3
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
z = torch.randn(2, 2)
u = torch.randn(2, 2)
mm = torch.randn(2, 2)
