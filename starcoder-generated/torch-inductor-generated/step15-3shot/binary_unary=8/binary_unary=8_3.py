
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = torch.dropout(v1, 1 - 0.3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 128, 128)
