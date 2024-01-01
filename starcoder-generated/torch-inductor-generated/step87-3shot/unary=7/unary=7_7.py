
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = self.linear(x)
        z = y * torch.clamp(torch.add(y, 3), min=0, max=6)
        w = z / 6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)

# Generating a trace
traced_cell = torch.jit.trace(m, x)

# Serializing the trace
traced_cell._save_for_lite_interpreter('model.ptl')

