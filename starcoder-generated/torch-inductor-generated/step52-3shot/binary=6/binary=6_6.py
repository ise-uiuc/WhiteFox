
class Model(torch.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(19, 19)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - other
        return v2


# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(20, 19)
