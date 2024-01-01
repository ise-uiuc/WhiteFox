
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3033, 79, bias=True)
        self.fc1_4 = torch.nn.functional.relu6

    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return self.fc1_4(v5)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.tensor([[0.2342, 2.534234, 1.22344, 34.4325345]], dtype=torch.float32)
