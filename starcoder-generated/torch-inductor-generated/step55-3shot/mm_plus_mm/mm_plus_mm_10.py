
class Model(torch.nn.Module):
    def forward(self, input1, input2, input5, input6):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input5, input6)
        t3 = torch.mm(input1, input2)
        t4 = torch.mm(input5, input6)
        return t1 + t2 + t3 + t4
# Inputs to the model
input1 = torch.randn(6, 6)
input2 = torch.randn(6, 6)
input5 = torch.randn(6, 6)
input6 = torch.randn(6, 6)
