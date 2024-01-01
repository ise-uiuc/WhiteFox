
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(200, 300)
        self.w2 = torch.nn.Linear(300, 5)
        self.dropout = torch.nn.Dropout(p=0.5)
 
    def forward(self, x, y):
        v1 = self.dropout(torch.nn.functional.softmax(self.w1(x)))
        v2 = self.dropout(torch.nn.functional.softmax(self.w2(v1)))
        res = y[:, :, 0].sum() + y[:, :, 1].sum() * 2 + y[:, :, 2].sum() * 3 + v2
        return res
 
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(10, 200)
y = torch.randn(10, 5, 3)
