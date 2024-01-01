
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
other = torch.randn(16)
if torch.cuda.is_available():
    m = m.to(torch.device(type='cuda', index=0))
    x1 = x1.to(torch.device(type='cuda', index=0))
    other = other.to(torch.device(type='cuda', index=0))
