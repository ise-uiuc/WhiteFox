
class Model(torch.nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(in_features=6, out_features=128, bias=True)
        self.ln = nn.LayerNorm((128, -1))
        self.linear2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.linear3 = nn.Linear(in_features=64, out_features=16, bias=True)
        super().__init__()
    def forward(self, x):
        out = self.linear1(x)
        out = self.ln(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.linear3(out)
        return out

# Inputs to the model
x = torch.randn(1, 2, 3, 4)

