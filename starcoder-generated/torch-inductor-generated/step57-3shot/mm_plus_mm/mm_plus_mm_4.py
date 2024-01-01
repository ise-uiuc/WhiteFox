
class Model(torch.nn.Module):
    def forward(self, input1, input2, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input2)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(16, 3)
input2 = torch.randn(16, 7)
input4 = torch.randn(3, 7)
