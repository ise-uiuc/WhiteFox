
class Model(torch.nn.Module):
    def forward(self, input):
        matrix = torch.nn.Parameter(torch.randn(1024, 1000))
        output = torch.mm(input, matrix)
        return output
# Inputs to the model
input = torch.randn(5, 15)
