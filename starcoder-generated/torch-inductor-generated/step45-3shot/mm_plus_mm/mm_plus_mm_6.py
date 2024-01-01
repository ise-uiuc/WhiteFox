
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        return input1.mm(input2) * input3.mm(input1)
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
