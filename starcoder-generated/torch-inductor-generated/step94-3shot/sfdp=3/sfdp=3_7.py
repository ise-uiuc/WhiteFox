
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 10)
        self.fc2 = torch.nn.Linear(10, 20)
        self.fc3 = torch.nn.Linear(20, 30)
        self.fc4 = torch.nn.Linear(30, 40)
 
    def forward(self, x1):
        v1 = self.fc1(x1) # Apply the first linear transformation
        v2 = self.fc2(v1) # Apply the second linear transformation
        v3 = self.fc3(v2) # Apply the third linear transformation
        v4 = self.fc4(v3) # Apply the forth linear transformation
        return v4

print("")
print("*** Input shape of test input tensor for the model is the following:")
# Inputs to the model
x1 = torch.randn(10, 5)
print(x1.shape)

