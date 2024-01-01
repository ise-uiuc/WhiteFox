
def model(t1):
    t2 = fc(t1)
    t3 = torch.clamp_min(t2, min_value=0)
    t4 = torch.clamp_max(t3, max_value=1)
    return t4

# Initializing the model
