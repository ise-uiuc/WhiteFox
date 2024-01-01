
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        # No forward path, only inference
        return torch.ones((1, 2, 2))
# Inputs to the model
