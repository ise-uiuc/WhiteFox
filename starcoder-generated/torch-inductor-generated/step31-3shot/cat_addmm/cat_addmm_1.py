
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 1)
        self.stack1 = torch.stack
        self.stack2 = torch.stack
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stack1(x)
        x = self.layers(x)
        x = self.stack2(x)
        return x
# Inputs to the model
x = torch.randn(4, 2)
