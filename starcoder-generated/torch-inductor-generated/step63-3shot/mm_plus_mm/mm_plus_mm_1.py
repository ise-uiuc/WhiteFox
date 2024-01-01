
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        output1 = torch.mm(input1, input2)
        output2 = torch.mm(input3, input4)
        return output1 + output2
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
