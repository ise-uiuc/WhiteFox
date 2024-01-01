
class Model(torch.nn.Module):

    def __init__(self, linear_transformation_1_min_value: float = -1.0, linear_transformation_1_max_value: float = 0.0):
        super().__init__()
        self.linear_transformation_1_min_value = linear_transformation_1_min_value
        self.linear_transformation_1_max_value = linear_transformation_1_max_value
        self.linear = torch.nn.Linear(123, 8)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min=self.linear_transformation_1_min_value)
        v3 = torch.clamp_max(v2, max=self.linear_transformation_1_max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(123, 1)
