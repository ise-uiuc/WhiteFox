
class Model(torch.nn.Module):
    def __init__(self, d_in=16, d_out=32, dropout=0.2):
        super().__init__()
        self.linear = torch.nn.Linear(d_in, d_out)
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return self.dropout(v3)

# Initializing the model
m = Model()
# Inputs to the model
x1 = torch.randn(1, 16)
