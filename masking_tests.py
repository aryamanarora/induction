from main import *
from experiments import *

loss, good_induction_candidates, tokenizer, toks_int_values = construct_circuit()
ds = Dataset({"toks_int_var": toks_int_values})
eval_settings = ExperimentEvalSettings(device_dtype=DEVICE, run_on_all=False)
corr = make_experiments()["unscrubbed"]
exp = Experiment(loss, ds, corr, 1)
scrubbed_circuit = exp.scrub()
inps = get_inputs_from_model(scrubbed_circuit.circuit)

ind_candidates_mask = get_induction_candidate_mask(inps[:, :-1], good_induction_candidates, match_all_occurrences=True)
pprint(list(zip(list(tokenizer.batch_decode(inps[0])), list(ind_candidates_mask[0]))))
