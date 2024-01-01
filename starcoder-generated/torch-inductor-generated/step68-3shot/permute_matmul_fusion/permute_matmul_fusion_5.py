
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 64) 
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 32)
        self.linear4 = torch.nn.Linear(64, 16)
    def forward(self, x1, x2):
        x1 = self.linear1(x1)
        x2 = self.linear1(x2)   

        x1 = self.linear2(torch.relu(x1))
        x2 = self.linear2(torch.relu(x2))  

        x1 = x1.expand(torch.squeeze(x1).size())
        x2 = x2.expand(torch.squeeze(x2).size())

        x1 = self.linear3(x1)
        x2 = self.linear3(x2)

        x1 = self.linear4(torch.relu(x1))
        x2 = self.linear4(torch.relu(x2))  

        x1 = torch.sigmoid(x1)
        x2 = torch.sigmoid(x2)  

        x1 = x1.permute(.75)
        x2 = x2.permute(.75)

        x2 = torch.bmm(x1, x2)

        return x2
# Inputs to the model
x1 = torch.randn(1, 8, 1, 1)
x2 = torch.randn(1, 8, 1, 1)
