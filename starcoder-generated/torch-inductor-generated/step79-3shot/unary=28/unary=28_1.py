 example: A model containing a linear transformation (in_features=`3`, out_features=`6`) with bias (size=`6`), and the output of the previous operation is clamped to a minimum value (`-1`) and maximum value (`1`).

# Initializing the model
m = torch.nn.Linear(in_features='3', out_features='6', bias='True') 
m = torch.nn.Linear(3, 6, bias=True) 
print(m.weight) # Weight tensor (3 * 6)
m.bias.data = m.bias.data - 1 # Set bias to -1 to meet requirement t2 = torch.clamp_min(t1, min_value=-1) 
 
# Inputs to the model
x1 = torch.randn(1, 3)
