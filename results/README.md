All files in this directory apart from this one should be automatically
generated by running an experiment with the --save flag set to True.

Each file has the name format {exp_name}.pkl.

These unpickle into (res, inp_ixes, meta), where res is the model's
num_samples x seq_len tensor of losses, inp_ixes is the 1d num_samples long
tensor of dataset indices the model was evaluated on, and meta is a dictionary
containing information about the experiment run.