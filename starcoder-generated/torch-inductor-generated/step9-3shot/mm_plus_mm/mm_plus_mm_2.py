
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        x1 = torch.mm(input1, input2)
        x2 = torch.mm(input3, input1)
        x3 = x1 + x2
        return x3
# Inputs to the model
input1 = torch.randn(6, 6)
input2 = torch.randn(6, 6)
input3 = torch.randn(6, 6)
