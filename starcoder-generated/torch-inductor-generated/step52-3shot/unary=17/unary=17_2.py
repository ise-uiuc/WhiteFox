
class Reshape(torch.nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
        self.fc = torch.nn.Linear(12, 8)

    def forward(self, x1):
        x1 = x1.view(-1, 2, 3, 2)
        return self.fc(x1)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Generate the model with the API of your choice. The pattern that trigger the bug should appear in the original
        self.Reshape = Reshape()
    def forward(self, x1):
        v1 = self.Reshape(x1)
        return x1 + v1
# Inputs to the model
x1 = torch.randn(5, 48)
