
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10, bias=True)
        self.min_value = 3.33
        self.max_value = 5.55
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=self.min_value)
        v3 = torch.clamp_max(v2, max_value=self.max_value)
        return v3

# Initializing the model
m = Model()

