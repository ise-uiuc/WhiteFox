
class SRN(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear = torch.nn.Linear(state_dim, action_dim)
 
    def forward(self, s):
        a = self.linear(s)
        a = a + self.linear(s)
        return a

# Initializing the model.
m = SRN(action_dim = 8, state_dim = 8)

# Inputs to the model
s = torch.randn(1, 8)
