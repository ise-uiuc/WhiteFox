
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, t1, min_value=0.25, max_value=0.5):
        t2 = self.linear(t1)
        t3 = torch.clamp(t2, min=min_value)
        return torch.clamp(t3, max=max_value)

# Initializing the model
m = Model()

