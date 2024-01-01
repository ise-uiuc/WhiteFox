
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):  # Add more inputs here to increase the number of inputs to this model.
        return torch.mm(torch.mm(input1, input2), input3) + input3

# Your code starts here
# Inputs to the model
input1 = # Your input here.
input2 = -100000
input3 = -100000
input4 = -100000
# Please change the name of the class to "ExampleModel"
class ExampleModel(torch.nn.Module)
    pass
# Your code ends here
