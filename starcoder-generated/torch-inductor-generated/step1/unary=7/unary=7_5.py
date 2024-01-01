
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = torch.nn.Linear(15, 8)
        self.f2 = torch.nn.Linear(10, 12)

    def forward(self, input0, input1):
        v1 = torch.mul(self.f1(input0), torch.clamp_max(torch.clamp_min(self.f2(input1)+3, min=0), max=6))
        v2 = v1 / 6
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randint((17, 10), 0, 10, dtype=torch.int32)
y = torch.randint((8, 20), 0, 10, dtype=torch.int32)
