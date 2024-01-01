
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        v1 = torch.addmm(input1, input2, input3)
        v2 = torch.mm(input2, input4)
        v3 = torch.mm(input1, input2)
        output = torch.log(torch.sum(v1, dim=1) \
            * torch.sum(v2, dim=1) * torch.sum(v3, dim=1))
        return output
# Inputs to the model
input1 = torch.randn(1, 1000)
input2 = torch.randn(1000, 1000)
input3 = torch.randn(1000, 1000)
input4 = torch.randn(1000, 1000)
