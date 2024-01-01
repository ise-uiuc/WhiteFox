
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 10)
 
    def forward(self, x3):
        t2 = torch.clamp_min(t1, min_value=0)
        t3 = torch.clamp_max(t2, max_value=2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x3 = np.array()
