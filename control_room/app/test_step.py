from well_models.fowm_model import load_initial_state, load_params, run_fowm_step

params = load_params()
state = load_initial_state()
print("x0:", [round(v, 2) for v in state])
for i in range(5):
    state, value = run_fowm_step(state, 2.0, params)
    print(f"step {i}: pdg={value:.1f} Pa, state[0]={state[0]:.2f}")
