
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, 0.0)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)

# Config to use for model compilation
model_config = {
    "input_info": {
        "x1": {
            "sample_size": [1, 8, 8], 
            "dtype": "float32"
        }
    },
    "output_info": {
        "output0": {
            "sample_size": [1, 8, 8], 
            "dtype": "float32"
        }
    },
    "fallback_device": {
        "device_type": "cpu"
    }
}

# Function to compile and perform optimization
def run_model(gm):
    gm = gm.eliminate_dead_code() # Eliminate dead code before compiling model
    gm.add_pass(pass_name="common::dead_code_elimination")
    opt_level = 2 # Set optimization level
    model_name = "dropout" # Set model name
    model, params, model_library_format = gm.compile(model_name=model_name, model_config=model_config, optimization_level=opt_level, disable_nhwc_to_nchw=True)
    # Run inference on CPU
    torch.set_default_tensor_type(torch.FloatTensor)
    # Run inference on GPU
    #torch.set_default_tensor_type(torch.cuda.FloatTensor)                                                                                        
    for _ in range(10): # Repeat inference for 10 times
        result = model(x1)

# Original model
with gm.create_graph() as gm_ori:
    # Begin code to generate original model
    # End code to generate original model
    run_model(gm_ori)

# Optimized model 
# The code below will be used later in the tutorial, you can customize it as you like
with gm.create_graph() as gm_opt:
    # Begin code to generate optimized model
    # End code to generate optimized model   
    run_model(gm_opt)


# Verify the optimized model 
optimized_node_list = gm_opt.graph.get_nodes_by_optype("lowmem_dropout")
if(len(optimized_node_list) == 0):
    raise ValueError(
        "The generated model uses the PyTorch function lowmem_dropout. Please generate new models that use this PyTorch function."
    )
if(len(optimized_node_list) > 1):
    raise ValueError(
        "Generated model has multiple nodes that use the PyTorch function lowmem_dropout. Please generate new models that use this PyTorch function only once."
    )
    
optimized_node_list = gm_opt.graph.get_nodes_by_optype("rand_like")
if(len(optimized_node_list) == 0):
    raise ValueError(
        "The generated model uses the PyTorch function rand_like. Please generate new models that use this PyTorch function."
    )
if(len(optimized_node_list) > 1):
    raise ValueError(
        "Generated model has multiple nodes that use the PyTorch function rand_like. Please generate new models that use this PyTorch function only once."
    )

# Model passes verification
# The generated model should contain dropout, but cannot replace the rand_like function.

