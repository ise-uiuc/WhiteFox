
class Model(torch.nn.Module):
    def forward(self, input):
        input = input + 1
        for i in range(100):
            input = torch.mm(input, input)
        return input
# Inputs to the model
input = torch.randn(200, 200)
