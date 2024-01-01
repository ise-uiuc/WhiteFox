
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(22, 2)
        self.relu1 = torch.nn.ReLU()
        self.layer_norm1 = torch.nn.LayerNorm([22])
        self.dense1 = torch.nn.Linear(2, 3)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.tanh1 = torch.nn.Tanh()
    def forward(self, a1):
        q2 = self.linear1(a1)
        q4 = self.relu1(q2)
        q5 = self.layer_norm1(q2)
        q3 = self.dense1(q4)
        q1 = self.dropout1(q2)
        q1 = self.tanh1(q2)
        return q1
# Inputs to the model
x1 = torch.randn(1, 22)
