
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x3 = torch.nn.functional.relu(x1)
        x4 = F.dropout(x1, p=0.5)
        return x4
# Inputs to the model
x1 = torch.randn(4,5,6,7)
