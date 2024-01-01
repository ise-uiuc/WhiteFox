
class Model(torch.nn.Module):
    def forward(self, input):
        b1 = torch.mm(input, input)
        b2 = torch.mm(input, input)
        b3 = torch.mm(input, input)
        b4 = torch.mm(input, input)
        b5 = torch.mm(input, input)
        c1 = torch.mm(b1, input)
        return b1 * b2 + c1

# Inputs to the model
input = torch.randn(5, 5)
