from task4.methods.vanilla import Vanilla


# GCSC = the Vanilla recipe with exactly one change: RandAugment(num_ops=2, magnitude=9)
# in the training transform (see configs/gcsc.yaml -> train_transform: gcsc).
# Optimization, init, schedule, epochs, seed, and checkpoint rule are identical.
class GCSC(Vanilla):
    pass
