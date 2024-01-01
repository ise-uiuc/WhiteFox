
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, input):
        x = self.batch_norm(input)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x
# Inputs to the model
input = torch.randn(1, 1000)
