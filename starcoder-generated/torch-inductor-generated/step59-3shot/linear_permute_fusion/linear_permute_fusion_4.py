
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, input):
        a1 = self.linear.bias
        a0 = input + a1
        a2 = torch.nn.functional.linear(a0, self.linear.weight)
        return a2
# Inputs to the model
input = torch.randn(1, 1, 2, device='cpu')
