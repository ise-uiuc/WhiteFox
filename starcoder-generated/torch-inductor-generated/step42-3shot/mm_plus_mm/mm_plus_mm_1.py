
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, x, w, y, z):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = t1 + t2
        t4 = torch.mm(input3, input4)
        return t4 + t3 + x + w + y + z
# Inputs to the model
input1 = torch.randn(3, 4)
input2 = torch.randn(4, 3)
input3 = torch.randn(3, 4)
input4 = torch.randn(4, 3)
x = torch.randn(1)
w = torch.randn(1)
y = torch.randn(1)
z = torch.randn(1)
