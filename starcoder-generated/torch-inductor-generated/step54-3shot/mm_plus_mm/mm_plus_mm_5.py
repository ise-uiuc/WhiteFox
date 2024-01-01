
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        a2 = torch.mm(input1, input4)
        b2 = torch.mm(input2, input3)
        z = a2 - b2
        t1 = torch.mm(input1, input4)
        t2 = torch.mm(input2, input3)
        return t1 * z
# Inputs to the model
x = torch.randn(4, 4)
y = torch.randn(4, 4)
