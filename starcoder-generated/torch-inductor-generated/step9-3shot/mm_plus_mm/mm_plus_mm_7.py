
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input2, input4)
        t3 = t1 + t2
        return t3
# Inputs to the model
input1 = torch.randn(7, 7, dtype=torch.float64)
input2 = torch.randn(7, 7, dtype=torch.float64)
input3 = torch.randn(7, 7, dtype=torch.float64)
input4 = torch.randn(7, 7, dtype=torch.float64)
