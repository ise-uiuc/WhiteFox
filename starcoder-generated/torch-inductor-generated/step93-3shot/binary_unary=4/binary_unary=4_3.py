
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return torch.relu(v2)

# Initializing the model
m = Model()

# For PyTorch models, we also need to specify inputs for dynamic output shapes.
m.eval()
dummy_input = torch.randn(1, 8)
m = torch.utils.mobile_optimizer.optimize_for_mobile(m)
torch._C._jit_pass_insert_debug_logic(m.graph)
m._save_for_lite_interpreter("./model.ptl")

# Input to the model (not necessary for PyTorch models)
inputx1 = torch.randn(1, 8)

