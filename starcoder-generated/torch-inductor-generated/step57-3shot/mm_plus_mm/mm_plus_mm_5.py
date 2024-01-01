
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input2, input3)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(7, 7)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
