
class Model(nn.Module):
    def forward(self, input):
        t1 = (input + input).mm(input + input)
        return t1
# Inputs to the model
input = torch.randn(5, 5)
