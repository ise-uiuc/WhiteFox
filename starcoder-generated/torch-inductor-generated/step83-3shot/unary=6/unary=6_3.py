
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(input_tensor)
        v2 = v1 + 3 # Adding 3
        v3 = torch.clamp(v2, min=0) # Clamping min
        v4 = torch.clamp(v3, max=6) # Clamping max
        v5 = v1 * v4 # Multiplying tensors
        v6 = v5 / 6 # Dividing by constant
        return v6
# Input to the model
x1 = torch.randn(1, 3, 128, 128)
