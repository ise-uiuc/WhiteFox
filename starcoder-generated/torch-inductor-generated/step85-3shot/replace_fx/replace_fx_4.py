
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        x = inputs + inputs
        y = x * x
        z1 = torch.randn(1, 2, 2)
        outputs = (z1, z1)
        return inputs
inputs = torch.randn(2, 2)
model = Model()
model_traced = torch.jit.trace(model, (inputs, ))
# Inputs to the model
inputs = torch.randn(2, 2)
