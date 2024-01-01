
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(p=torch.rand(()))
        self.dropout2 = torch.nn.Dropout(p=torch.rand(()))
    def forward(self, x1):
        x2 = torch.unsqueeze(torch.randn((), requires_grad=True), 0)
        x2 = self.dropout1(x2)
        x2 = self.dropout2(x2)
        x2 = x2 * 2
        return x2 + x1
# Inputs to the model
x1 = torch.randn(-1, requires_grad=True)
