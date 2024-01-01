
class Model(nn.Module):
    def forward(self, input):
        v1 = torch.mm(input, input)
        v2 = torch.mm(input, input)
        v3 = torch.mm(input, input)
        v3 = input.mm(input)

        x = torch.mm(input, input)
        v4 = x.mm(input)
        return v1 + v2 + v3 + v4
# Inputs to the model
input = torch.randn(10, 10)
