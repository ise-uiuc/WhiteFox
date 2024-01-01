
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        self.x = torch.randn(3, 516)
        v1 = self.x + x
        v2 = torch.rand_like(self.x)
        x3 = v1 + v2
        return x3

# Input
x = torch.randn(516, 3)
# Model Ends
model = Model()
model.eval()
pred = model(x)

# The final output shape of the model should be (3, 516)

