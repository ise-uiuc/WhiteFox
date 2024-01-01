
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)

    def forward(self, input):
        output = self.linear1(input)
        output = self.linear2(output)
        output = self.linear3(output)
        return output
# Inputs to the model
input = torch.randn(1, 32)
