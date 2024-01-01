
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 10
        input_size = 8
        self.linear = torch.nn.Linear(input_size, num_classes)
 
    def forward(self, x):
        fc = self.linear(x)
        fc_clamp = torch.clamp(fc, min=0.0)
        fc_clamp_clamp_max = torch.clamp_max(fc_clamp, 0.99999)
        return fc_clamp_clamp_max

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
