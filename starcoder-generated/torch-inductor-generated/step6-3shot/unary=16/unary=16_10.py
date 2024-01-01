
import torch
class CustomModel(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        
        # Initialize one linear transformation in the model
        self.linear = torch.nn.Linear(input_channel, output_channel)
    
    # Define the forward behavior
    def forward(self, x1):
        linear_output = self.linear(x1)
        relu_output = torch.relu(linear_output)
        
        # You may also use pre-defined PyTorch functions to save development cost
        return relu_output

# Initializing the model
m = CustomModel(3, 8)

# Inputs to the model
x1 = torch.randn(1, 3)
