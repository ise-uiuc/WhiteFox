
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10, False)
 
        # Initialization values used for creating this module.
        # Please note that there are no restrictions on how these values are initialized.
        self.weight_init_values = torch.tensor([0.65838746, 0.58118068, 0.57962785, 0.74222197, 0.88190979,
                                                0.81708118, 0.93021691, 0.68442443, 0.73882091, 0.0325654 ])
 
    def forward(self, x1):
        self.linear.weight.data[0] = self.weight_init_values[0]
        self.linear.weight.data[1] = self.weight_init_values[1]
        self.linear.weight.data[2] = self.weight_init_values[2]
        self.linear.weight.data[3] = self.weight_init_values[3]
        self.linear.weight.data[4] = self.weight_init_values[4]
        self.linear.weight.data[5] = self.weight_init_values[5]
        self.linear.weight.data[6] = self.weight_init_values[6]
        self.linear.weight.data[7] = self.weight_init_values[7]
        self.linear.weight.data[8] = self.weight_init_values[8]
        self.linear.weight.data[9] = self.weight_init_values[9]
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v3 = v1 + (torch.pow(v1, 3)) * 0.044715
        v4 = v3 * 0.7978845608028654
        v5 = torch.tanh(v4)
        v6 = v5 + 1
        v7 = v2 * v6
        return v7

# Initializing the model
m = CustomModel()

# Inputs to the model
x1 = torch.randn(1, 5, 100)
