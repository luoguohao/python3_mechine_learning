import hyperopt
from hyperopt import hp
from hyperopt import fmin, tpe


def objective(args):
    case, val = args
    return val
    # if case == 'case 1':
    #     return val
    # else:
    #     return val ** 2


# define a search space
space = hp.choice('a',
                  [
                      ('case 1', 1 + hp.uniform('c1', 0, 1)),
                      # ('case 2', hp.uniform('c2', -10, 10))
                      ('case 2', hp.uniform('c2', 0, 1))
                  ])

# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=5000)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(hyperopt.space_eval(space, best))
# -> ('case 2', 0.01420615366247227}
