
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=2, out_features=4)
        self.linear2 = torch.nn.Linear(in_features=2, out_features=4)
        self.linear3 = torch.nn.Linear(in_features=2, out_features=3)
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        y = torch.cat([x, x], dim=3)  # Sink cat after linear2
        y = self.linear3(y)
        return y
# Inputs to the model
x = torch.randn(2, 2)
