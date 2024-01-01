
# PyTorch model with the required pattern is found. It is listed here for your convenience:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(11, 15, 7, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.797884
class PyTorchModule:
    def __init__(self, model, example_inputs):
        self.model = model
        self.example_inputs = example_inputs
pytorch_model = PyTorchModule(Model(), torch.randn(1, 11, 3097, 193))
