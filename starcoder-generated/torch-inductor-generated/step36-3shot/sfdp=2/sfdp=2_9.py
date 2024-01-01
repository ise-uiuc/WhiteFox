
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 16)
 
    def forward(self, x1):
         q = self.fc1(x1)
         k = self.fc2(x1)
         v = self.fc3(x1)
 
         temp = torch.matmul(q, k.transpose(1, 0))
         temp2 = temp.transpose(1,0)
         a = torch.softmax(10 * temp2, -1)
         temp1 = torch.matmul(a, v)
         return temp1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(12, 16)
