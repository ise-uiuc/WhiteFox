
class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(44, 15)
  def forward(self, x5):
    v1 = self.fc(x5)
    v2 = torch.sigmoid(v1) * 0.5
    v3 = torch.sigmoid(v2) * torch.sigmoid(v1)
    v4 = v3 * v1
    v5 = v4 + 0.044715
    v6 = v5 * 0.7978845608
    v7 = torch.tanh(v6)
    v8 = v7 * v7
    v9 = v8 + v7
    v10 = v9 * v9
    v11 = torch.tanh(v10)
    v12 = v11 + 1
    v13 = v2 * v12
    return v13
# Inputs to the model
x5 = torch.randn(23, 44)
