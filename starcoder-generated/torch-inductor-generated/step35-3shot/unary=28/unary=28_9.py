
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=self.min_value)
        v3 = torch.clamp_max(v2, max=self.max_value)
        return v3

# Initializing the model
min_value = 0.1
max_value = 0.5
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.tensor([[[-float("inf"), -float("inf"), -float("inf")], [-float("inf"), -float("inf"), -float("inf")], [0.13429049, 0.42054142, 0.49232574]], [[-float("inf"), -float("inf"), -float("inf")], [-float("inf"), -float("inf"), -float("inf")], [0.16724844, 0.54488579, 0.13837028]], [[0.37899566, 0.49897517, 0.57421803], [0.04109592, 0.0537002, 0.10894445], [0.49481009, 0.2592683, 0.41894564]], [[0.56204269, 0.08512115, 0.33184175], [0.55831719, 0.54522006, 0.53325084], [0.43586658, 0.3994147, 0.40781801]], [[0.15681072, 0.30640938, 0.3510543], [0.36911216, 0.20468537, 0.33771543], [0.36799981, 0.35813716, 0.19680074]], [[-0.01074977, 0.55161162, 0.0129508], [0.57406411, 0.25938863, 0.31956928], [0.53489292, 0.20629274, 0.53235047]], [[0.58171878, 0.50436866, 0.33519348], [0.53819465, 0.55675534, 0.06166232], [0.31104006, 0.14132397, 0.50913526]]], dtype=torch.float32)
