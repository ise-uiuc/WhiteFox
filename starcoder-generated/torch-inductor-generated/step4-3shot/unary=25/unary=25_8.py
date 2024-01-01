
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        m1 = v1 > 0
        v2 = v1 * 0.2
        v3 = torch.where(m1, v1, v2)
        return v3
__init__
v1 = torch.zeros(1, 8)
with torch.no_grad():
    m1 = v1 > 0
    v2 = v1 + 1
    v3 = m1.bool()
torch.where(m1, v1, v2)
with torch.no_grad():
    m2 = v1 > 2.4
with torch.no_grad():
    m3 = v3 | m2
v4 = m1.float()
v3[m3] = v4[m3] * v2[m3] + (v1 - v2)[m3] * v3[m3]
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
