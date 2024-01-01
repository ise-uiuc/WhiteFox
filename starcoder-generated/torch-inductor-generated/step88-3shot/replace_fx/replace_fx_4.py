
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.modules.activation.Dropout(p=0.5)
        self.dropout2 = torch.nn.modules.activation.Dropout(p=0.5)
        self.dropout3 = torch.nn.modules.activation.Dropout(p=0.5)
        self.dropout4 = torch.nn.modules.activation.Dropout(p=0.5)
    def forward(self, x1):
        x2 = self.dropout1(x1 + 1.0)
        x3 = self.dropout2(torch.rand_like(x2) + 1.0)
        x4 = self.dropout3(torch.rand_like(x2) + 1.0)
        x5 = self.dropout4(torch.rand_like(x2) + 1.0)
        x6 = x2 * x3 * x4 + x5
        return x6
# Inputs to the model
x1 = torch.randn(1, 2, 3)
