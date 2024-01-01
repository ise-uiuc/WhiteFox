
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        res = torch.mm(input1, input1)
        res += torch.mm(input2, input4)
        res += torch.mm(input3, input3)
        res += torch.mm(input3, input2)
        return res
# Inputs to the model
input1 = torch.randn(200, 200)
input2 = torch.randn(200, 200)
input3 = torch.randn(200, 200)
input4 = torch.randn(200, 200)
