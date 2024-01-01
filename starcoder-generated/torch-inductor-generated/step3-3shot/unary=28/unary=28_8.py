
class Model(torch.nn.Module):
    def __init__(self, min_value=1.0, max_value=2.0):
        super().__init__()

    def forward(self, x1):
        v1 = x1
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
__import_statement__ = "import torch"
min_value = random.random()
max_value = random.random()
parameters = {
    "min_value": min_value,
    "max_value": max_value,
}
m = Model(**parameters)

# Inputs to the model
__input_0_shapes__ = [(1, 32, 16, 5)]
x1 = __call__(torch.randn(1, 32, 16, 5))
