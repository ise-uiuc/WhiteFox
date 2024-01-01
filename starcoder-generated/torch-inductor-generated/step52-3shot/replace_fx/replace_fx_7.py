
# Note: dropout with p = 1 should result in a functionally equivalent result to dropout with p = 0
class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=1.0)
        return x2
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()    
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.0)
        return x2
# Inputs to the model
data = torch.randn(1, 2, 2)
