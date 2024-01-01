
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 5)
        self.fc3 = nn.Linear(5, 4)
   
    def forward(self, x):
        x = F.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return x

# Initialize the model
m = Net()

# Inputs to the model
x = torch.randn(5, 2)
