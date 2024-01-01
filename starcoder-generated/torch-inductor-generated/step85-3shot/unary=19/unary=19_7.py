
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)

    def forward(self, x1):
        v5 = self.linear(x1)
        v4 = torch.sigmoid(v5)
        return v4
# Initializing the model
m = Model()

#Inputs to the model
x1 = torch.randn(1, 3)
