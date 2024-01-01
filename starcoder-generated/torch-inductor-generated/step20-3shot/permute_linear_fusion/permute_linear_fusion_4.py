
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input1 = torch.nn.Linear(3, 3)
        self.input2 = torch.nn.Linear(3, 2)
    def forward(self, input):
        v1 = input.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.input1.weight, self.input1.bias)
        x = v2.permute(1, 2, 0)
        v3 = x.contiguous()
        return torch.relu(v3) + self.input2(v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
