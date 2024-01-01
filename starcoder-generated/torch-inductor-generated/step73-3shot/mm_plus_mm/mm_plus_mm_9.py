
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        x1 = torch.mm(input1, input1)
        x2 = torch.mm(input2, input2)
        x3 = torch.mm(input1, input1)
        return (x2 + -x2) * (2.0 / 10) * (x1 + -x1) * x3 * 2
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 2)
