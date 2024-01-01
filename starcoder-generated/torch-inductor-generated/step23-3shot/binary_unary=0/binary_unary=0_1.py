
x = torch.randn(1, 16, 64, 64, requires_grad=True)
model = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0) # Define model
y = model(x) # Use model
y.sum().backward(retain_graph=True) # Compute gradient of y
z = y.detach() # Detach y
a = torch.relu(z) # Apply the relu activation
a.sum().backward() # Compute gradient of a
# Inputs to the model
x = torch.randn(1, 16, 64, 64, requires_grad=True)
