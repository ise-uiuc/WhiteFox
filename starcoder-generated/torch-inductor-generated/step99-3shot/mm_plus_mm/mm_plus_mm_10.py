
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        out = input1 * input2 * input3
        out = out + input4
        return out
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
