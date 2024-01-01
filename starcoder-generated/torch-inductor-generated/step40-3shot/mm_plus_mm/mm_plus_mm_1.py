
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)

    def forward(self, input):
        return self.linear1(self.linear2(input))
# Inputs to the model
input = torch.randn(8, 4)
