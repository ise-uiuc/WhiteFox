
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(
            v1, kwargs[f'min_{kwargs["value_type"]}'], 
            inplace=kwargs['inplace']
        )
        v3 = torch.clamp_max(
            v2, kwargs[f'max_{kwargs["value_type"]}'], 
            inplace=kwargs['inplace']
        )
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
kwargs = {
    "value_type": "tensor",
    "min_tensor": x2,
    "max_tensor": x2,
    "inplace": True
}
