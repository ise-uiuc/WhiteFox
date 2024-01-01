
class Model(torch.nn.Module):
    def __init__(self, min, max, input_size):
        super().__init__()
        self.input_size = input_size
        self.seq = torch.nn.Sequential(torch.nn.Linear(input_size[0], input_size[1]), torch.nn.Sigmoid())
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.seq(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.7
max = 1.4
input_size = [101,1]
# Inputs to the model
x1 = torch.randn(100, input_size)
