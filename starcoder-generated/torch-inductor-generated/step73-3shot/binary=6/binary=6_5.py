
class Model(torch.nn.Module):
    def __init__(self, out):
        super().__init__()
        self.linear = torch.nn.Linear(1, out)
     
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.7515788566919324
        return v2

# Initializing the model
m = Model(5)

# Inputs to the model
x1 = torch.empty(2, 1).uniform_(0, 1)
