
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = torch.clamp(x1, min=655.36, max=-655.36)
        return torch.nn.functional.linear(v1, torch.nn.Linear(1, 1).weight, torch.nn.Linear(1, 1).bias)
# Inputs to the model
x1 = torch.randn(1, 1)
