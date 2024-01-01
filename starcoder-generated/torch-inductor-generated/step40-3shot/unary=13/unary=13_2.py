
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32, bias=True)
        self.linear2 = torch.nn.Linear(32, 64, bias=True)
 
    def forward(self, q1):
        y1 = self.linear1(q1)
        y2 = torch.sigmoid(y1)
        y3 = y1 * y2
        output = self.linear2(y3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 64, 1)
output = m(q1)

