
class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Linear(3, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 3),
            torch.nn.ReLU(),
        )
        self.min_value, self.max_value = -1000, 0

    def clamp(self, x):
        return x.clamp(self.min_value, self.max_value)
 
    def forward(self, x1):
        x2 = self.clamp(self.clamp(self.linear1_0(x1)))
        return self.clamp(self.clamp(self.linear1_3(x2)))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
