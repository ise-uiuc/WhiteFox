
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x):
        v1 = x.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, None)
        return torch.sum(torch.nn.functional.hardtanh(torch.nn.functional.tanh(v2), -1., 1.))
# Inputs to the model
x = torch.randn(1, 2, 2)
