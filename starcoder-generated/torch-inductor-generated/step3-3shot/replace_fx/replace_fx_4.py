
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = torch.nn.functional.dropout(torch.ones(5))
    def forward(self, x1):
        return torch.nn.functional.dropout(self.q1)
# Inputs to the model
x1 = torch.randn(1)
