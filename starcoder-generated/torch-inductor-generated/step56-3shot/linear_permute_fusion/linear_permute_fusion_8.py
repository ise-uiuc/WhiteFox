
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, input):
        v7 = self.linear.weight
        v0 = torch.nn.functional.linear(input, v7)
        v3 = input.permute(0, 2, 1)
        v1 = v0.permute(0, 2, 1)
        v2 = v1 * v7
        v6 = input.shape
        return v2
# Inputs to the model
input = torch.randn(1, 2, 2, device='cpu')
