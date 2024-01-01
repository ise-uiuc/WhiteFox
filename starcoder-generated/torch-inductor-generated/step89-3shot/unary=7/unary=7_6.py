
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.op = torch.nn.Linear(6, 3)
 
    def forward(self, x1):
        v70 = x1 + 3.0
        v71 = torch.unsqueeze(v70, 0)
        v72 = torch.unsqueeze(v71, 2)
        v73 = v72
        v74 = torch.zeros(1, 6, 1, dtype=torch.float32, device='cpu')
        v75 = torch.where(v74 >= 0.0, v73, v74)
        v76 = torch.where(v74 < 0.0, v74, v73)
        v77 = v75 * v76
        v78 = v77 + 3.0
        v79 = torch.floor(v78 / 6.0)
        v80 = torch.floor(v79)
        v81 = v80
        v82 = torch.tensor(-1, dtype=torch.float32)
        v83 = torch.max(v82, v81)
        v84 = torch.tensor(6, dtype=torch.float32)
        v85 = torch.min(v84, v83)
        v86 = v78 - (6.0 * v85)
        v87 = v86 / 6.0
        v88 = torch.unsqueeze(v87, 0)
        v89 = torch.unsqueeze(v88, 2)
        v90 = v89
        v91 = torch.zeros(1, 6, 1, dtype=torch.float32, device='cpu')
        v92 = torch.where(v91 >= 0.0, v90, v91)
        v93 = torch.where(v91 < 0.0, v90, v91)
        v94 = v92 * v93
        v95 = v94 + 3.0
        v96 = torch.floor(v95 / 6.0)
        v97 = torch.floor(v96)
        v98 = v97
        v99 = torch.tensor(-1, dtype=torch.float32)
        v100 = torch.max(v99, v98)
        v101 = torch.tensor(6, dtype=torch.float32)
        v102 = torch.min(v101, v100)
        v103 = v95 - (6.0 * v102)
        v104 = v103 / 6.0
        v105 = torch.unsqueeze(v104, 0)
        v106 = torch.unsqueeze(v105, 2)
        v107 = v106
        v108 = torch.zeros(1, 6, 1, dtype=torch.float32, device='cpu')
        v109 = torch.where(v108 >= 0.0, v107, v108)
        v110 = torch.where(v108 < 0.0, v107, v108)
        v111 = v109 * v110
        l1 = self.op(v111)
        l2 = l1 * (torch.clamp(l1 + 3, min=0, max=6))
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
