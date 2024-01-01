
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x


# Model starts
class ModelTanh(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = torch.tanh(x)
        return x


# Inputs to the model
x = torch.randn(1, 3, 64, 64)
#