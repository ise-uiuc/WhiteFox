
class Model(torch.nn.Module):
    def forward(self, input1, input2, input5, input6):
        t1 = torch.mm(input1, input2)
        return t1 + t1
# Inputs to the model
input1 = torch.randn(32, 32)
input2 = torch.randn(32, 32)
input5 = torch.randn(32, 32)
input6 = torch.randn(32, 32)
