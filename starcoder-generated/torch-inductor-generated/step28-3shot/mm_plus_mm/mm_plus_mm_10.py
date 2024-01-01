
class Model(nn.Module):
    def forward(self, input1, input2, input3, input4):
        t = torch.mm(input1, input4)
        return input3 + input1
# Inputs to the model
input1 = torch.randn(1)
input2 = torch.randn(1)
input3 = torch.randn(1)
input4 = torch.randn(1)
