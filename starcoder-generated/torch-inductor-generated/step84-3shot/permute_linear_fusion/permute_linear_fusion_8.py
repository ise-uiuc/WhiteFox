
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = self.linear(x1)
        x2 = torch.nn.functional.interpolate(v1, size=(2, 2), mode='bilinear')
        x3 = torch.nn.functional.relu(x2)
        y = self.linear(x3) + x1
        return x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
