
class TorchMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(self.input_size, 1024, )
        self.act = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(1024, 1)
        self.a_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.b_dropout = torch.nn.Dropout(p=1 - self.dropout_rate)
    def forward(self, x):
        x = self.a_dropout(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.b_dropout(x)
        return self.linear2(x)
# Inputs to the model
self.dropout_rate = 0.5
self.input_size = 1024
x1 = torch.randn(16, 1024)
