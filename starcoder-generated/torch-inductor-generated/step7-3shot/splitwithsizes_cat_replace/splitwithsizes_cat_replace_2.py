
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1):
        t1 = torch.split(x1, [4, 1, 2, 5], dim=1)[0]
        t2 = torch.split(x1, [4, 1, 2, 5], dim=1)[1]
        t3 = torch.split(x1, [4, 1, 2, 5], dim=1)[2]
        t4 = torch.split(x1, [4, 1, 2, 5], dim=1)[3]
        t5 = torch.cat([t1, t3], dim=1)
        t6 = torch.cat([t2, t4], dim=1)
        t7 = torch.cat([t5, t6], dim=1)
        t8 = torch.cat([t3, t3], dim=1)
        return torch.cat([t2, t8], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
torch.onnx.export(m, x1, 'tmp.onnx', opset_version=11) # Export the model to ONNX for correct results.
m(x1) # Output of the model

is_valid_splitwithsizes_cat(m) # Whether the current model meets the requirements or not


