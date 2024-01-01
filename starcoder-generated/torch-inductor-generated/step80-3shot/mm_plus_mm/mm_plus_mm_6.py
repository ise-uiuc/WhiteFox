
class Model(torch.nn.Module):
    def forward(self, input):
        torch.mm(input, input)
        torch.mm(input, input)
        torch.mm(input, input)
        torch.mm(input, input)
        torch.mm(input, input)
        return input + input
# Inputs to the model
input = torch.randn(10, 10)
