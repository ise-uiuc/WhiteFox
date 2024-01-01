
class model(torch.nn.Module):
    def __init__(self, input_tensor):
        super().__init__()
        self.input_tensor = torch.rand_like(input_tensor)
    def forward(self, input_tensor):
        out = self.input_tensor
        return out
# Inputs to the model
x1 = torch.randn(1, 2, 2)
