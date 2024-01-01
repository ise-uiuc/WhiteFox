
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        m = torch.nn.functional.max_pool1d(input, 1)
        return torch.squeeze(m, dim=2)
# Inputs to the model
input = torch.randn(2, 2, 10)
