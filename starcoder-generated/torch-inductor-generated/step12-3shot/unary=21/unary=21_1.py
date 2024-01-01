
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conva = nn.Conv2d(1, 32, 3, stride=3, padding=2)
        self.activation = nn.Tanh()
    def forward(self, data):
        x = self.conva(data)
        x = self.activation(x)
        return x
# Inputs to the model
data = torch.randn(2, 1, 448, 448)
