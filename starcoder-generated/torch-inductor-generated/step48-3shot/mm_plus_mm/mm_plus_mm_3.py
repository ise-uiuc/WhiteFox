
class Model(torch.nn.Module):
    def forward(self, input):
        v1 = torch.mm(input, input)
        v2 = torch.mm(input, input)
        v3 = input.mm(input)
        v4 = input.mm(input)
        v5 = torch.mm(input, input)
        return v1 + v2
# Inputs to the model
x = torch.randn(2, 2)
# Input to the model
