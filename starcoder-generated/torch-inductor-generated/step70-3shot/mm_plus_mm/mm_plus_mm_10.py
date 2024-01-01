
class Model(torch.nn.Module):
    def forward(self, input5, input6):
        t1 = torch.mm(input5, input6)
        t2 = torch.mm(input5, input6)
        t3 = torch.mm(input5, input6)
        t3 = torch.mm(input5, input6)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 4)
input5 = torch.randn(4, 4)
input6 = torch.randn(4, 4)
