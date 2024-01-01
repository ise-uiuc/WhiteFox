
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_ = torch.nn.functional.relu_
    def forward(self, x1):
        v1 = self.relu_(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 5)
