
class Model(torch.nn.Module):
    def forward(self, input):
        a = torch.mm(input, input)
        b = torch.mm(input, input)
        c = a + b
        return c
# Inputs to the model
input = torch.randn(5, 5)
