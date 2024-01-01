
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()

        self.min_value = min_value
        self.max_value = max_value
     
    
    def forward(self, x1):
        v1 = x1.reshape(x1.size(0), 64 * 64 * 3)
        v2 = torch.nn.functional.linear(v1, torch.ones(16369))
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
min_value = 136
max_value = 8438
m = Model(min_value, max_value)
x1 = torch.zeros(2, 3, 64, 64)
