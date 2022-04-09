from itertools import product
from base import BaseGenerator

class MetaphorGenerator(BaseGenerator):

    reasoning_type = 'metaphor'

    objects = {'ball', 'dax', 'widget', 'card'}
    agents = {'girl', 'woman', 'boy', 'man', 'dog', 'cat'}

    def gen_motion_metaphors(self):
        for obj in self.objects:
            premise = f"The {obj} is flying."

            yield ( premise, f"The {obj} is moving.", 'entailment' )
            yield ( premise, f"The {obj} is moving fast.", 'entailment' )
            yield ( premise, f"The {obj} is moving quickly.", 'entailment' )

            yield ( premise, f"The {obj} is moving slowly.", 'contradiction' )
            yield ( premise, f"The {obj} is not moving.", 'contradiction' ) # ?

    def gen_orientation_metaphors(self):
        for agent in self.agents:
            premise = f"The {agent} is feeling down."
            yield ( premise, f"The {agent} is unhappy.", 'entailment' )
            yield ( premise, f"The {agent} is happy.", 'contradiction' )


    # TODO: 
    #
    # she's close to resigning -> she's in contact with resigning
    # she's close to 
    #
    # he drank himself out of the promotion -> he was in the promotion
    #
    # He got out out of the chores -> he was in the chores
    # get got out of bed -> he was in bed