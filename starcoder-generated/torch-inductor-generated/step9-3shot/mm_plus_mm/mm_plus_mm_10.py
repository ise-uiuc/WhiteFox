
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        torch.arange()
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
input4 = torch.randn(8, 8)
