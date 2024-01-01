
class Model(torch.nn.Module):
    def forward(self, input):
        x = torch.mm(input, input)
        y = torch.mm(x, input)
        return y
# Inputs to the model
input = torch.randn(2, 2)
