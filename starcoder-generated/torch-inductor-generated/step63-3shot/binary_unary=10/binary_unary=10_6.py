
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
     # Creating a linear transformation operation
    def init_weights(self):
        self.weight = torch.randn(512, 1024) * math.sqrt(
            (1.0 / 512)
        )  # Xavier initialization for the linear layer
        self.bias = torch.zeros(
            1024
        )  # Initializing bias as constant and can be filled using zeros/random/whatever methods
 
    # Defining the forward function of the model
    def forward(self, x1):
        v1 = torch.addmm(self.bias, x1, self.weight.t())
        v2 = torch.nn.functional.relu(
            v1
        )  # The ReLU function is applied to the output of the linear transformation
        return v2

# Initializing the model
m = Model()

# Calling the init_weights function
m.init_weights()
x1 = torch.randn(3, 512)
