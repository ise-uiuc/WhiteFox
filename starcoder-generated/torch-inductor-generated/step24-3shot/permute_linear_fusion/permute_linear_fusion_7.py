
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        x2 = torch.relu(torch.tanh(x1))
        x2 = x2.squeeze(dim=-1)
        x3 = x2.permute(0, 2, 1)
        x3 = x3.squeeze(dim=-1)
        return torch.nn.functional.softmax(torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias), dim=-1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
