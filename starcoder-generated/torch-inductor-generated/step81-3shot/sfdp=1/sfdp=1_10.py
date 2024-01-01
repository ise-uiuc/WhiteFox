
class Model(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias
 
    def forward(self, x1):
        x2 = torch.matmul(x1, self.weight.transpose(-2, -1))
        x3 = x2.div(196)
        x4 = torch.nn.functional.softmax(x3, dim=-1)
        x5 = torch.nn.functional.dropout(x4, 0.7)
        x6 = torch.matmul(x5, self.bias)
        return x6

# Initializing the model
m = Model(torch.randn(1, 16, 30, 64), torch.randn(1, 16, 64))

# Inputs to the model
x1 = torch.randn(1, 16, 60, 64)
