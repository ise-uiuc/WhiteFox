
class Model(torch.nn.Module):
    def forward(self, weight, bias, inp1, inp2):
        weight = torch.mm(weight, bias)
        o = torch.matmul(inp1, weight)
        t = o * inp2
        return t + inp1
# Inputs to the model
weight = torch.rand(3, 3)
bias = torch.rand(3, 3)
inp1 = torch.rand(3, 5)
inp2 = torch.rand(5, 5)
