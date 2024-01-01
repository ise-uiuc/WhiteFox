
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.clamp_min(v1, min_value=-100.0)
        v3 = torch.clamp_max(v2, max_value=100.0)
        return v3
   
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
