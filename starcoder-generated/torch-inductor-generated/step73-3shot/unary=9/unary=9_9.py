
class Model(torch.nn.Conv2d):
    def forward(self, x1):
        x2 = 3 + super().forward(x1)
        x3 = torch.clamp(x2, min=0)
        x4 = torch.clamp(x3, max=6)
        x5 = torch.div(x4, 6)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
