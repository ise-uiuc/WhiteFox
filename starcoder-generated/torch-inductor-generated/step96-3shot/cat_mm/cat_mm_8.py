
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        result = []
        t1 = torch.mm(input1, input2)
        result.append(t1)
        t12 = torch.mm(input2, input3)
        result.append(t12)
        t123 = torch.mm(input3, input1)
        result.append(t123)
        t1234 = torch.mm(input3, input1)
        result.append(t1234)
        t12345 = torch.mm(input3, input1)
        result.append(t12345)
        return torch.cat(result, 1)
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
input4 = torch.randn(3, 3)
