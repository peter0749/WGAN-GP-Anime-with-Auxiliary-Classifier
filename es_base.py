import numpy as np

class ES(object):
    def __init__(self, fitness=lambda x: (1-x[...,0])**2 + 100*(x[...,1]-x[...,0]**2)**2, 
                    dna_length=2, bound=[-50,50], generations = 300, population_size=100, offspring_size=50, type='minimize'):
        self.F = fitness
        self.dna_n = dna_length
        self.gen = generations
        self.bound = bound
        self.pop_n = population_size
        self.kid_n = offspring_size
        self.f_sign = -1 if type=='minimize' else 1
    def initialization(self, mean=0.0, std_dev=2.0, std_dev_m=1.0):
        pop = dict(DNA=std_dev * (mean + np.random.randn(self.pop_n, self.dna_n)),
               mut_strength=std_dev_m * np.random.randn(self.pop_n, self.dna_n))
        return pop
    def get_fitness(self, pred):
        assert type(pred)==np.ndarray, 'You must pass a np.ndarray'
        return self.f_sign * self.F(pred).flatten()
    def get_offspring(self,pop):
        kids = {'DNA': np.empty((self.kid_n, self.dna_n))}
        kids['mut_strength'] = np.empty_like(kids['DNA'])
        for kv, ks in zip(kids['DNA'], kids['mut_strength']):
            p1, p2 = np.random.choice(np.arange(self.pop_n), size=2, replace=False)
            cp = np.random.randint(0, 2, self.dna_n, dtype=np.bool)  # crossover points
            kv[cp] = pop['DNA'][p1, cp]
            kv[~cp] = pop['DNA'][p2, ~cp]
            ks[cp] = pop['mut_strength'][p1, cp]
            ks[~cp] = pop['mut_strength'][p2, ~cp]
            
            # mutate (change DNA based on normal distribution)
            ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
            kv += ks * np.random.randn(*kv.shape)
            kv[:] = np.clip(kv, *self.bound)    # clip the mutated value
        return kids
    def put_kids(self, pop, kids):
        for key in ['DNA', 'mut_strength']:
            pop[key] = np.vstack((pop[key], kids[key]))
        return pop
    def selection(self, pop):
        fitness = self.get_fitness(pop['DNA'])
        idx = np.arange(pop['DNA'].shape[0])
        good_idx = idx[fitness.argsort()][-self.pop_n:]
        for key in ['DNA', 'mut_strength']:
            pop[key] = pop[key][good_idx]
        return pop