
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        x3 = torch.rand_like(x2) # Replace this function
        x4 = F.dropout(x2) # Replace this function
        return x4 # Eliminate this node
# Inputs to the model
x1 = torch.randn(1, 2, 2)
