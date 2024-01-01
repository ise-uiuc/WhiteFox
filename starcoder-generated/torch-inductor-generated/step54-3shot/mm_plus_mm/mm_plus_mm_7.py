
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        y = torch.bmm(input1, input2)
        return y
# Inputs to the model
input1 = torch.randn(4, 3, 50)
input2 = torch.randn(4, 50, 3)
