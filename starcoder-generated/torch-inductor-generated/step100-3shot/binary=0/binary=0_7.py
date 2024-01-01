
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.linear = torch.nn.Linear(4, 4)
    def forward(self, x, w1, b1, w2, bias=True, bias2=torch.tensor([-1.5])):
        # Add layers here
        # Use shape inference
        # Specify non-default parameters
        # Specify different non-default arguments
        return x
# Input to the model
x = torch.randn(1, 3, 107, 199)
