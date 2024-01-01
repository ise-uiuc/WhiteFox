
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        v1 = input1.mm(input2)
        v2 = input3.mm(input4)
        v3 = v1 + v2
        r = torch.mm(v3, input4)
        return r
# Inputs to the model
input1 = torch.randn(2, 10)
input2 = torch.randn(20, 2)
input3 = torch.randn(2, 20)
input4 = torch.randn(21, 2)
