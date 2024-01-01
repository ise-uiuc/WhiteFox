
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 1)
 
    def forward(self, input_tensor):
        intermediate_variable_1 = self.linear(input_tensor)
        intermediate_variable_2 = intermediate_variable_1 * 0.5
        intermediate_variable_3 = intermediate_variable_1 * 0.7071067811865476
        intermediate_variable_4 = torch.erf(intermediate_variable_3)
        intermediate_variable_5 = intermediate_variable_4 + 1
        intermediate_variable_6 = intermediate_variable_2 * intermediate_variable_5
        return intermediate_variable_6

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn((1, 5))
