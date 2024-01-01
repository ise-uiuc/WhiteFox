
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 3)

        self.fc2 = torch.nn.Linear(3, 4)

    def forward(self, x1, x2):
        x3 = torch.cat((x1, x2), dim=0)
        y1 = self.fc1(x3)

        # Add layer fc1 to the module dictionary
        self.add_module('fc1', self.fc1)

        y2 = self.fc2(y1)

        # Add layer fc2 to the module dictionary
        self.add_module('fc2', self.fc2)

        return y3
# Inputs to the model
x1, x2 = torch.randn(1, 2), torch.randn(1, 2)
