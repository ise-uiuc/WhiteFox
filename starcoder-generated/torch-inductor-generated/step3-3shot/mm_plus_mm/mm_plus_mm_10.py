
def some_function(input1, input2, input3, input4):
    t1 = torch.mm(input1, input2)
    t2 = torch.mm(input3, input4)
    t3 = t1 + t2
    return t3
# Create model
model = torch.jit.script(some_function)
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
input4 = torch.randn(3, 3)
