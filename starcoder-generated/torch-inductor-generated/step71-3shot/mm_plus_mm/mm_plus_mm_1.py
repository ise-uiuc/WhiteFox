
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(3, 1)
    def forward(self, x):
        output = self.linear1(x)
        return output
# Inputs to the model
x = torch.randn(1, 3)
