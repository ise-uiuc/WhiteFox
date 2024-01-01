
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.rand_like(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
