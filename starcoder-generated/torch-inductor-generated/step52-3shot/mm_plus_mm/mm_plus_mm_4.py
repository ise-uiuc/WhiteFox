
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t11 = torch.mm(input1, input3)
        t12 = torch.mm(input3, input1)
        t13 = t11 + t12
        return t13
# Inputs to the model
input1 = torch.randn(15, 15)
input2 = torch.randn(15, 15)
input3 = torch.randn(15, 15)
