
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(6, 8)
        self.linear2 = torch.nn.Linear(8, 10)
 
    def forward(self, x):
        h = torch.tanh(self.linear1(x))
        y = torch.flatten(self.linear2(h), 1)
        return y

# Initializing the model
