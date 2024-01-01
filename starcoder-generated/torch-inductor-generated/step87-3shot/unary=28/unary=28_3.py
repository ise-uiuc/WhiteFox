
class Model(torch.nn.Module):
    def __init__(self, min_value=-17, max_value=17):
        super(Model, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        v1 = torch.sum(x)
        v2 = v1 + self.min_value
        v3 = v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(512)
