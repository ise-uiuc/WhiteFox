
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.sign(v2)
        v3 = torch.min(v2, dim=-1)[1]
        x2 = torch.min(v3, dim=-1)[1]
        x3 = x2.unsqueeze(dim=-1)
        v3 = v3 + x3.to(v3.dtype)
        v3 = torch.mean(v3.T)
        return (v1[0][0] == v3.item()).to(torch.float32)
# Inputs to the model
x1 = torch.randn(2, 2, 3)
# Model begins

# Inputs to the model
x1 = torch.tensor(
[[
[0.725322299194336, -0.1499673698425293, 0.18127508611679078],
[0.11947536118030548, 0.811821346282959, 0.6194396061897278]
],
[
[-0.3212654085159302, 0.710962381362915, 0.015438999503564835],
[0.3567740216255188, 0.8067044544219971, -0.5289223098754883]
]
])
# Model begins

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        pass
    def forward(self):
        m = torch.nn.Softmax()
        return m.weight
