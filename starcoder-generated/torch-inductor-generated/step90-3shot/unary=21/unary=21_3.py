
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, input):
        x = self.conv(input)
        x = 2. * x / (1. + torch.exp(-x)) - 1.
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.drop1(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.drop2(x)
        x = x / (1. + torch.exp(-x))
        x = 2. * x - 1.
        x = self.conv(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
input = torch.randn(1, 16, 10, 10)
