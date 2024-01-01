
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 2)
    
    def forward(self, x):
        x = F.dropout(F.gelu(self.layer1(x)))
        x = F.dropout(self.layer2(x), training=self.training)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
