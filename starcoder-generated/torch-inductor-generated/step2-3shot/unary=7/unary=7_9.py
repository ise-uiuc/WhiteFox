
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, input_tensor):
        out = self.linear(input_tensor)
        out = out * torch.clamp_min(torch.clamp_max(out + 3, 6), 0)
        out = out / 6
        return out

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 10)
