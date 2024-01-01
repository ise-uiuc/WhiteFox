
def linear(x1):
    t1 = x1.permute(0, 2, 1)
    v2 = torch.nn.functional.linear(t1, w0, b0)
    return v2

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v0 = self.linear.weight.T.unbind(0)
        w0_0 = v0[0]
        w0_1 = v0[1]
        b0 = self.linear.bias.unbind(0)
    	b0_0 = b0[0]
        b0_1 = b0[1]
        v1 = x1.permute(0, 2, 1)
        # v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        # v3 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v4 = linear(v1)
        v4 = v4.unsqueeze(-3)
        return v4 + v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
