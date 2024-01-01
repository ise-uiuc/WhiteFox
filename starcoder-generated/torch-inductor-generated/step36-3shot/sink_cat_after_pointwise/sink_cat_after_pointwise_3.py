
class InputReshape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.relu(x.view(2, 3))
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
