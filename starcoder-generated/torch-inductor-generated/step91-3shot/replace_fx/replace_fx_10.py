
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        F.dropout(x, training=True, inplace=True)
        F.dropout(x, training=False, inplace=True)
# Inputs to the model
x = torch.Tensor([[0.25, 0.25, 0.25, 0.25]])
