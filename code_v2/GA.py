# Code adapted from lab 8 and Conor Rothwell (https://www.linkedin.com/pulse/hyper-parameter-optimisation-using-genetic-algorithms-conor-rothwell/)

from LoadTrainTest import prepare_data, train_casper, validate_casper, test_casper
from deap import algorithms, base, creator, tools
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import time


num_classes = 3
input_size = 25
# Prepare data for input into Casper Model/s
test_data, train_dataset, val_dataset = prepare_data()

# Used to calculate time
time_start = time.time()


# Initialise GA
toolbox = base.Toolbox()
# Minimal loss as fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define Hyper-parameter bounds
lower_batch_size, upper_batch_size = 1, 50
lower_p, upper_p = 1, 50
lower_epoch, upper_epoch = 100, 1000
lower_l1, upper_l1 = 0.05, 0.5
lower_l1l2, upper_l1l2 = 0.01, 0.05
lower_l2l3, upper_l2l3 = 0.1, 0.5

# Define initialisation of genes
toolbox.register("attr_batch_size", random.randint, lower_batch_size, upper_batch_size)
toolbox.register("attr_p", random.randint, lower_p, upper_p)
toolbox.register("attr_epoch", random.randint, lower_epoch, upper_epoch)
toolbox.register("attr_l1", random.uniform, lower_l1, upper_l1)
toolbox.register("attr_l1l2", random.uniform, lower_l1l2, upper_l1l2)
toolbox.register("attr_l2l3", random.uniform, lower_l2l3, upper_l2l3)

N_CYCLES = 1
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_batch_size, toolbox.attr_p, toolbox.attr_epoch,
                  toolbox.attr_l1, toolbox.attr_l1l2, toolbox.attr_l2l3), n=N_CYCLES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

'''
Individual Evaluation function.
Train Casper network and use Validation Accuracy as individual evaluation
Returns validation accuracy
'''
def evaluation(individual):
    # Hyper-parameters
    batch_size = individual[0]
    P = individual[1]
    num_epochs = individual[2]
    l1 = individual[3]
    l2 = l1*individual[4]
    l3 = l2*individual[5]

    # Train w/ params 3 times and get the average validation loss
    total_loss = 0
    average_over = 3
    for i in range(average_over):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        net = train_casper(train_loader, num_classes, input_size, P, num_epochs, l1, l2, l3)
        val_loss = validate_casper(net, val_dataset, input_size)
        total_loss += val_loss

    return total_loss/3,

'''
Mutation function.
If mutation is to occur, randomly change a gene to a value within its bounds
Returns individual with mutated gene
'''
def mutate(individual):
    gene = random.randint(0, 5)  # select which parameter to mutate
    if gene == 0:
        individual[gene] = random.randint(lower_batch_size, upper_batch_size)
    elif gene == 1:
        individual[gene] = random.randint(lower_p, upper_p)
    elif gene == 2:
        individual[gene] = random.randint(lower_epoch, upper_epoch)
    elif gene == 3:
        individual[gene] = random.uniform(lower_l1, upper_l1)
    elif gene == 4:
        individual[gene] = random.uniform(lower_l1l2, upper_l1l2)
    elif gene == 5:
        individual[gene] = random.uniform(lower_l2l3, upper_l2l3)

    return individual,


# One point cross over
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", mutate)
toolbox.register("evaluate", evaluation)
toolbox.register("select", tools.selTournament, tournsize=3)

# Record statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# GA hyper-parameters
crossover_probability = 0.8
mutation_probability = 0.2
generations = 2
pop = toolbox.population(n=3)

result, log = algorithms.eaSimple(pop, toolbox,
                            cxpb=crossover_probability, mutpb=mutation_probability,
                            ngen=generations, verbose=False,
                            stats=stats)

# Get best parameters
best_parameters = tools.selBest(result, k=1)[0]

# Record time taken
time_end = time.time()

# Unpack best parameters
batch_size = best_parameters[0]
P = best_parameters[1]
num_epochs = best_parameters[2]
l1 = best_parameters[3]
l2 = l1*best_parameters[4]
l3 = l2*best_parameters[5]

# Test best parameters
total_acc = 0
average_over = 3
for i in range(average_over):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    net = train_casper(train_loader, num_classes, input_size, P, num_epochs, l1, l2, l3)
    test_acc = test_casper(net, test_data, input_size)
    total_acc += test_acc

# Average Testing Accuracy
avg_acc = total_acc/average_over

print(best_parameters)
print("Test accuracy average:", avg_acc, "%")
print("Seconds elapsed: ", time_end-time_start)


plt.figure(figsize=(11, 4))
plots = plt.plot(log.select('min'),'c-', log.select('avg'), 'b-', log.select('max'), 'r-')
plt.legend(plots, ('Minimum Loss', 'Mean Loss', 'Maximum Loss'), frameon=True)
plt.ylabel('Validation Loss'); plt.xlabel('Generations');
plt.show()

