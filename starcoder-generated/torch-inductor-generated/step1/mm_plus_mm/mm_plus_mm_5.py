
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = torch.nn.Linear(10, 6)
    self.linear2 = torch.nn.Linear(10, 6)
 
  def forward(self, x, y):
    v1 = self.linear1(x)
    v1 = self.linear2(v1)
    v2 = v1 * 0.7
    v3 = v1 * 1.3
    v4 = torch.erf(v3)
    v5 = v4 + 0.12
    v6 = x * 2.1
    v6 = v1 * v5
    v7 = y * 0.2
    v8 = v2 * v7
    v9 = v7 * 0.3
    v10 = v2 * v9
    v11 = v10 + v8
    v12 = v2 * v10
    v13 = v1 * v6
    v13 = v8 + v13
    v14 = v11 + v12
    v15 = v14 * v8
    v16 = v13 * v12
    v17 = v15 * v8
    v18 = v15 * v10
    v19 = v11 * v10
    v20 = v11 * v9
    v21 = v13 * v6
    v22 = v9 * v14
    v22 = v22 * v12
    v23 = v22 * v8
    v24 = v18 + v19
    v25 = v16 + v24
    v26 = v21 + v25
    v27 = v23 + v17
    v28 = v26 * v23
    v29 = v26 * v18
    v30 = v19 * v9
    v31 = v16 * v17
    v32 = v11 + v13
    v32 = v10 + v22
    v33 = v20 + v30
    v34 = v29 + v31
    v35 = v32 * v28
    v36 = v26 + v33
    v36 = v34 + v36
    v37 = v34 * v36
    v38 = v28 * v33
    v39 = v29 * v29
    v40 = v27 * v21
    v41 = v40 * v23
    v42 = v41 * v30
    v43 = v42 * v29
    v44 = v38 * v33
    v45 = v39 * v36
    v46 = v36 * v35
    v47 = v45 * v30
    v48 = v37 * v13
    v49 = (v47 + v38) * v48
    v50 = (v44 + v38) * v27
    v51 = v43 * v42
    v52 = v42 * v21
    v53 = v52 * v33
    v54 = v53 * v30
    v55 = v37 * v35
    v56 = v31 * v52
    v57 = (v56 + v37) * v28
    v58 = v47 + v46
    v58 = v46 + v58
    v59 = v51 + v54
    v60 = v51 + v57
    v61 = v49 + v50
    v62 = v59 + v60
    v63 = v62 * v28
    return v63

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
y = torch.randn(1, 10)
