
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 10, bias=False)

    def forward(self, input):
        v1 = self.linear(input)
        v2 = v1 - 1
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()
input = torch.randn(1, 1)

