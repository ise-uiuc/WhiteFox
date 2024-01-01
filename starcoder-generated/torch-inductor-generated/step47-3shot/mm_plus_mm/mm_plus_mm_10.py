
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        a = torch.mm(input1, input1)
        b = torch.mm(input2, input2)
        c = torch.mm(input2, input1)
        return a * b - c
# Inputs to the model
input1 = torch.randn(7, 4)
input2 = torch.randn(7, 4)
