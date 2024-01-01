
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(4,)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * torch.clamp(0, 6, l1 + 3)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn((1, 4))
output = m(x1)
print("Done!" if torch.allclose(output, torch.Tensor([-0.62463525, -0.33384856, 0.14618831, 0.49907909]), atol=1e-06) else "Failed!")
