
class Model(torch.nn.Module):
    def forward(self, input):
        return torch.mm(input, input)
# Inputs to the model
input = torch.randn(7, 7, 7)
