
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        v1 = torch.mm(input1, input2)
        t1 = torch.mm(input3, input4)
        t2 = v1 + t1
        t4 = torch.mm(input5, input5)
        return t1 + t2 + t4
# Inputs to the model
input1 = torch.randn(32, 32)
input2 = torch.randn(32, 32)
input3 = torch.randn(32, 32)
input4 = torch.randn(32, 32)
input5 = torch.randn(32, 32)
