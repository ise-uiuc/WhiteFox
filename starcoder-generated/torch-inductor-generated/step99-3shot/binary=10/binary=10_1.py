
class Model(torch.nn.Module):
    def forward(self, __input__):
        return torch.nn.functional.linear(__input__, torch.nn.Parameter(torch.randn([16, 10]))) + __input__

# Initializing the model
m = Model()

# Inputs to the model
