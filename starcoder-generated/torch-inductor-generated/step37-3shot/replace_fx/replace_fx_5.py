
class Model(torch.nn.Module):
    def forward(self, x):
        x = x + 2
        x = x + 2
        x = x + 2
        x = torch.nn.functional.dropout(x)
        return x
# Inputs to the model
x = torch.ones((2, 2))
