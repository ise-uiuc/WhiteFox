
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.47703261193942216)
    def forward(self, x1):
        v1 = self.dropout(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
