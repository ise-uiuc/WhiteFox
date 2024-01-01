
# This example only use 2D but any number of input tensors are accepted
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(2, 1, 0)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.arange(4) + 1
        v3 = v3.unsqueeze(1)
        v4 = torch.reshape(v3, (2, 2))
        v5 = v2[: v4.size()[0], -v4.size()[1] :]
        return v5
# Inputs to the model
x1 = torch.linspace(1,9,9).view(3,3).to(torch.float32)
