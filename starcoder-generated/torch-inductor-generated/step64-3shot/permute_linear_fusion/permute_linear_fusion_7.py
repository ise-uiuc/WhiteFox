
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.relu1 = torch.nn.ReLU()
        self.relu6 = torch.nn.ReLU6(inplace=False)
    def forward(self, x1):
        # TBD
        return (self.relu1(x), self.relu6(x))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
