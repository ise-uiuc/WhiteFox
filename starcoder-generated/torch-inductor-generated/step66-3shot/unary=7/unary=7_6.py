
class Model2_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = y1 * torch.clamp(y1 + 3, min=0, max=6) 
        y3 = y2 / 6
        return y3

# Initializing the model
m = Model2_model()

# Inputs to the model
x1 = torch.randn(1, 128)
