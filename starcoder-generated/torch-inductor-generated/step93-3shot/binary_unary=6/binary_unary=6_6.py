
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64)

    def forward(self, x1, p=1.0):
        t1 = self.linear(x1)
        t2 = t1 - p
        t3 = torch.nn.functional.relu(t2)
        return t3

# Initializing and using the model
p = 2.4
m = Model()
