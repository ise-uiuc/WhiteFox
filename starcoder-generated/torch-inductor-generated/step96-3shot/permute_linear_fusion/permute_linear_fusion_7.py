
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1, 3, 4)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return torch.reshape(v2, (1, 2, 1, 2, 1))
# Inputs to the model
x1 = torch.tensor([[[[0.7195, 0.0788], [-1.5773, -0.7005]], [[0.8957, -2.0262], [0.5399, 0.7540]]]])
