
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.l1 = torch.nn.Linear(64, 64)
        self.l2 = torch.nn.Conv1d(64, 64, 1)
    def forward(self, x):
        logits = self.l1(x)
        t = torch.tanh(logits)
        t = self.l2(t)
        t = torch.tanh(t)
        return t
# Inputs to the model
x = torch.randn(1, 64)
