
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()      
    def forward(self, x):
        a1 = torch.nn.functional.dropout(x, False)
        a2 = torch.nn.functional.dropout(x, True)
        a3 = torch.nn.functional.dropout(x, True)
        a4 = torch.nn.functional.dropout(x, True)
        return a1
# Inputs to the model
x = torch.randn(1)
