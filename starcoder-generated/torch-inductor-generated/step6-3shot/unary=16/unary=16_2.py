
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the data set
dataset = Dataset()

# Initializing the model
m = Model()

# Initializing the optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(m.parameters(), lr=learning_rate, momentum=0.9)

# Number of epochs to update the weights
num_epochs = 30

# The training loop
for epoch in range(num_epochs):
    for i in range(0, 100):
        # Generate a random sample from the data set
        x1, label = dataset.next_sample()
    loss = model_loss(m(x1), label)
    optimizer.zero_grad()    
    loss.backward()
    optimizer.step()
    
x1 = torch.randn(1, 8)
