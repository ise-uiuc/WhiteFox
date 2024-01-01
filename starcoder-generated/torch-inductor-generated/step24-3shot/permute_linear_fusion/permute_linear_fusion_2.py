
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self):
        v1 = torch.randn(1, 2, 2)
        v2 = v1.transpose(0, 2)
        return v2
# Inputs to the model
