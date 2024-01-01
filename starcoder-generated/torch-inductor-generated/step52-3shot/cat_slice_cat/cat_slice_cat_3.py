
class Model(torch.nn.Module):
    self.linear1 = torch.nn.Linear(5, 10)
    self.linear2 = torch.nn.Linear(5, 10)
    self.linear3 = torch.nn.Linear(5, 10)
    self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1, x2, x3):
        y1 = torch.cat([self.linear1(x1), self.linear2(x2), self.linear3(x3)], dim=1)
        y2 = y1[:, 0:9223372036854775807]
        y3 = y2[:, 0:5]
        z = torch.cat([y1, y3], dim=1)
        return z

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 10)
x2 = torch.randn(1, 5)
x3 = torch.randn(1, 5, 10)
