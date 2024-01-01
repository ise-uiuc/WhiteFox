
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 64)
        self.linear1 = torch.nn.Linear(64, 100)
        self.linear2 = torch.nn.Linear(100, 400)
        self.linear3 = torch.nn.Linear(400, 9)

    def forward(self, x1):
        out = torch.nn.functional.softmax(x1,dim=-1)
        out1 = self.linear1(torch.nn.functional.relu(self.linear(out)))
        out1 = torch.nn.functional.dropout(out1, p=0.1, training=self.training)
        out2 = self.linear2(torch.nn.functional.tanh(out1))
        out3 = self.linear3(torch.nn.functional.relu(out2))
        return out3


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(2,500)
