
class Model(torch.nn.Module):
    def __init__(self, min_value=-1, max_value=10).to("cuda"):
        super().__init__()
        self.linear = torch.nn.Linear(8, 3)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x2 = torch.randn(1, 8, device="cuda")
