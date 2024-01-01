
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 4)
        self.linear = torch.nn.Linear(2, 4)
        self.linear_2 = torch.nn.Linear(4, 4)
        self.linear_3 = torch.nn.Linear(4, 2)
    def forward(self, x4):
        v0 = x4
        v1 = torch.nn.functional.relu(torch.nn.functional.linear(v0, self.linear_1.weight, self.linear_1.bias))
        v2 = torch.nn.functional.nn.functional.relu(torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)) + 1
        v3 = v2.permute(0, 2, 1)
        v4 = torch.nn.functional.relu(torch.nn.functional.linear(v3, self.linear_2.weight, self.linear_2.bias))
        output = []
        for k1 in range(v4.size(0)):
            v5 = torch.cat([v4[k1], v4[k1], v4[k1]], dim=0)
            output.append(v5)
        output = torch.stack(output)
        v6 = torch.cat([v4, v4, v4], dim=1)
        v7 = torch.cat([output, v6], dim=2)
        return torch.nn.functional.linear(v7, self.linear_3.weight, self.linear_3.bias)
# Inputs to the model
x4 = torch.randn(1, 2, 2)
