
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.reshape(x, (1, 1000, 1000))
        return torch.cat((y, y), dim=2)
    @torch.jit.export
    def run_pass(self, x):
        self.eval()
        with torch.no_grad():
            for _ in range(3):
                ret = self.forward(x)

        self.train()
        with torch.enable_grad():
            for _ in range(10):
                ret = self.forward(x)
# Inputs to the model
# This model takes one input tensor.
x = torch.randn(10, 10, 10)
