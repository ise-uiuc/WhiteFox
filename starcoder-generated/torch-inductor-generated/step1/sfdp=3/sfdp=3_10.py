
class Model(torch.nn.Module):
    def forward(self, x):
        x1 = x * x
        # TODO: Fill in the blanks
        x2 = self.dropout(x1)
        out = x2.matmul(value)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 8)
value = torch.randn(10, 1, 8)
