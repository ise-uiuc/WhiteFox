
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2) # Input size and hidden dimension of the layer 1
        self.linear2 = torch.nn.Linear(2, 2) # Hidden dimension of the layer 1 and hidden dimension of the layer 2
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 1)
        t1 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        out = t1 + x2
        return out
# Inputs to the model
w = 2;
x1 = torch.randn(2, 2, w, w)
x2 = torch.randn(2, 2, 2, 2)
