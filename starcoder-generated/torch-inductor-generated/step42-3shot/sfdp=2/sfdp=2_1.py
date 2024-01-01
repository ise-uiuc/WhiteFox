
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 8)
        self.linear3 = torch.nn.Linear(64, 8)
        self.linear4 = torch.nn.Linear(64, 8)


    def forward(self, x1, x2, x3):
        v1 = self.linear1(x1)
        v2 = self.linear3(x2)
        v3 = self.linear4(x3)
 
        v4 = torch.matmul(v1, v2.transpose(-2, -1))
        v5 = v4.div(0.5)
        v6 = v5.softmax(dim=-1)
        v7 = torch.nn.functional.dropout(v6, p=0.5, training=self.training)
        v8 = v3.matmul(v7)
        return v8
# # Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 64)
x3 = torch.randn(1, 64)
