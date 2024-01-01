
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1 - 1000
        v2 = F.relu(x1 - 250)
        return F.sigmoid(v1 - v2)
# Inputs to the model
x1 = torch.randn(3, 3, 227, 227)
