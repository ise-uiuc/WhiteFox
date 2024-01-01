
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        output = []
        for k1 in range(x1.size(0)):
            v3 = torch.cat([v2[k1], v2[k1], v2[k1]], dim=0)
            output.append(v3)
        output = torch.stack(output)
        return output
# Inputs to the model
x1 = torch.randn(1, 2, 2)
