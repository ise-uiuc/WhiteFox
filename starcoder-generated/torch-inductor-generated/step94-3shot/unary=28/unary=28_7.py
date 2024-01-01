
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_transform = torch.nn.Linear(16, 32)
        self.min_value = torch.nn.Parameter(torch.randn(()))
        self.max_value = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        v1 = self.linear_transform(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
num_batch = 1
x = torch.randn(num_batch, 16)
min_value = torch.min(x)
max_value = torch.max(x)
