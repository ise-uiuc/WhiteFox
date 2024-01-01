
class TestModel(torch.nn.Module):
    def forward(self, x):
        x = torch.mm(self.a, x)
        x = torch.mm(self.b, x)
        c = torch.mm(self.c, x)
        x = x + c
        x = c + x
        x = x + c
        x = c + x
        x = self.relu(x)
        x = self.relu(self.relu(self.relu(self.relu(x))))
        res = x * x
        return res
# Inputs to the model
input = torch.randn(4, 4)
