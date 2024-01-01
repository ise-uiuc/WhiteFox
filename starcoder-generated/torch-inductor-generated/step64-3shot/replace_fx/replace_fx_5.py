
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.25)
        self.dropout_no_args = torch.nn.Dropout(p=0.5)
    def forward(self, x2_3):
        x3_3 = self.dropout(x2_3)
        return x3_3
# Inputs to the model
x2_3 = torch.randn(1, 2)
