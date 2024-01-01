
model1 = Model()
model1.eval()

# Run the following code to get:
# `result` - the output of `model1(x)`, after the `fuse_conv_bn` optimization is applied
result = torch.jit.script(torch.jit.optimize_for_inference(model1))  
