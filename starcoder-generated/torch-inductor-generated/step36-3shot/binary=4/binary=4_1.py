
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__linear = torch.nn.Linear(64, 64) # No bias in the linear transform
        self.__other = torch.nn.Parameter(torch.randn(64, 64)) # The second tensor to be added to the output of the linear transformation
 
    def forward(self, x1):
        x2 = self.__linear(x1)
        x3 = x2 + self.__other
        return x3

# Initializing the model
m = Model()

# Optimizer and loss for training the model
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = torch.nn.Loss()

# Training the model
for i in range(training_steps):
    y_hat = m(x)
    loss = loss_fn(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

