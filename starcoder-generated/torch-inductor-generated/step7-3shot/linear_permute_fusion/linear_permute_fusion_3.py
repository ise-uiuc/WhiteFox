
class Model(torch.nn.Module):
    # Add a default dtype of float32 instead of float64
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, dtype=torch.float32)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = x1.view(v1.shape)
        return v2
# Inputs to the model
x1 = torch.randn(3, 7, 2)
