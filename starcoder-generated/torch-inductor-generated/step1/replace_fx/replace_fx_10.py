
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1):
        v1 = torch.nn.functional.dropout(x1, training=False)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(3, 4)
