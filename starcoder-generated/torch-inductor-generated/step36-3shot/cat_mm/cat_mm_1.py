
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        result = []
        result.append(x1)
        result.append(x1 + x1)
        result.append(torch.relu(x1 + x1))
        result.append(torch.relu(x1) + 0.1)
        result.append(torch.relu(x1) + 0.1)
        result.append(torch.relu(x1) + 0.1)
        result.append(torch.div(x1, 0.1))
        result.append(torch.mm(x1, x1) + 0.1)
        for loopVar1 in range(10 + 0):
            result.append(x1)
        return torch.cat(result, 1)
# Inputs to the model
x1 = torch.randn(10, 10)
