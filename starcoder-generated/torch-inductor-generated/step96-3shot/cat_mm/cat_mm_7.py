
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs1, inputs2, inputs3):
        results = inputs2 + inputs1
        results = results + inputs1 * inputs2
        results = results + 0.5 * inputs3
        return torch.cat([results, results, results], 1)
# Inputs to the model
inputs1 = 0.7 * torch.randn(8, 4)
inputs2 = 0.5 * torch.randn(8, 4)
inputs3 = 20 * torch.randn(8, 4)
