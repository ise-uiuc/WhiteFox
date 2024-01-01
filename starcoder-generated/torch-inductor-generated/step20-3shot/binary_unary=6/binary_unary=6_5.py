
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(25, 10)

    def forward(self, input_tensor):
        return self.linear(input_tensor) + 5.0

# Initializing the model
m = Model()

# input to the model
input_tensor = torch.randn(1, 25)
