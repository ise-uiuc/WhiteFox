
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose = torch.nn.Linear(4, 4)
    def forward(self, x):
        v1 = self.transpose(x)
        v2 = torch.nn.functional.linear(x, self.transpose.weight, self.transpose.bias)
        return v2+v1
# Inputs to the model
x1 = torch.randn(2,2,4, device='cpu')
