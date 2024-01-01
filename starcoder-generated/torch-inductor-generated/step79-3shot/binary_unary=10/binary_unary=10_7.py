
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(in_features=3, out_features=1)

    def forward(self, x1):
        y1 = self.fc(x1)
        y2 = y1 + other_tensor
        y3 = F.relu(y2)
        return y3 

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
