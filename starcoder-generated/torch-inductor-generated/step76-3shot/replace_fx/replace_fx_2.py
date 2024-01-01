
 class Model(torch.nn.Module):
     def __init__(self):
        super().__init__()
     def forward(self, x):
        a = F.dropout(x, p=0.5)
        b = F.dropout(x, p=0.5)
        return a + b
 # Inputs to the model
x = torch.randn(1, 2, 2)
