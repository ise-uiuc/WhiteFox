
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2)
    def forward(self, x):
        x = torch.nn.functional.dropout(self.layer1(x), p=0.2, training=True)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
