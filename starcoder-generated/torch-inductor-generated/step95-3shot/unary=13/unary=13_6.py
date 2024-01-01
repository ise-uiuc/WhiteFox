
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
        self.linear2 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        return torch.max(self.sigmoid(self.linear2(self.linear1(input))), dim=[2]).values


# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 3, 64, 64)
