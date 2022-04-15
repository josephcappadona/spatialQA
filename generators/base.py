from random import random

class BaseGenerator:

    reasoning_type = None
    ENTAILMENT = 'entailment'
    CONTRADICTION = 'contradiction'
    NEUTRAL = 'neutral'

    names = ['John', 'Jane', 'Tom', 'Alice', 'Bob', 'Sonya']

    def __init__(self, sample=1.0):
        self.sample = sample

    def generate_all(self):
        for attr in dir(self):
            if attr.startswith('gen_'):
                gen_fn = self.__getattribute__(attr)
                for p, h, e in gen_fn():
                    if random() < self.sample:
                        metadata = (self.reasoning_type, attr)
                        yield p, h, e, metadata
    
    def __iter__(self):
        for p, h, e, m in self.generate_all():
            yield (p, h), e, m