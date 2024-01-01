
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(20, 2)
        self.l2 = torch.nn.Linear(2, 2)

    def forward(self, x1):
        v1 = x1.view(-1, 20)
        v2 = self.l1(v1)
        v3 = torch.sigmoid(v2)
        output = self.l2(v3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(10, 20)
