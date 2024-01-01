
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bias = torch.nn.Parameter(torch.randn(1, 8, 1, 1), requires_grad=True)
 
    def forward(self, x, other):
        v1 = self.conv(x)
        v2 = v1 + self.bias
        v3 = F.relu(v2, **{"other": other})
        return v3

# Initializing all of the layers
conv_layer = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
x = torch.randn(1, 3, 64, 64)
x_copy = x.clone()
other = torch.randn(1, 8, 32, 32)
other_copy = other.clone()
bias = torch.nn.Parameter(torch.randn(1, 8, 1, 1), requires_grad=True)

# Check the original values
v1 = conv_layer(x)
v2 = v1 + bias
v3 = F.relu(v2, **{"other": other})
original_relu_values = v3.clone()
print(f"Max value: {torch.max(original_relu_values)}"
      f", Min value: {torch.min(original_relu_values)}")

# Modify the input of the ReLU layer
v1 = conv_layer(x_copy)
v2 = v1 + bias
v3 = F.relu(v2, **{"other": other_copy})
modified_relu_values = v3.clone()

# Show that the values remain the same after the ReLU operation
print(f"Max value: {torch.max(original_relu_values == modified_relu_values)}"
      f", Min value: {torch.min(original_relu_values == modified_relu_values)}")

# Check the gradient
v1.retain_grad()
v2.retain_grad()
v3.retain_grad()
v3.backward(torch.ones_like(v3))

conv_grad = v1.grad.clone()
add_grad = v2.grad.clone()
relu_grad = v3.grad.clone()


