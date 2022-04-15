from itertools import product
from .base import BaseGenerator


class MetaphorGenerator(BaseGenerator):

    reasoning_type = 'metaphor'

    objects = {'ball', 'dax', 'widget', 'card'}
    grounded_fliers = {'car', 'dog', 'horse', 'cat'}
    agents = {'girl', 'woman', 'boy', 'man', 'dog', 'cat'}
    events = {'movie', 'class', 'show', 'introduction'}
    jump_dests = {'to the next section', 'to the end of the movie', 'to conclusions'}

    def gen_motion_metaphors_skipping(self):
        for name, event in product(self.names, self.events):
            premise = f"{name} is skipping the {event}."

            yield ( premise, f"{name} is in motion.", "contradiction", 0 )
            yield ( premise, f"{name} is moving.", "contradiction", 1 )

    def gen_motion_metaphors_jumping(self):
        for name, jump_dest in product(self.names, self.jump_dests):
            premise = f"{name} is jumping {jump_dest}."

            yield ( premise, f"{name} is in motion.", "contradiction", 0 )
            yield ( premise, f"{name} is moving.", "contradiction", 1 )

    #### 

    def gen_motion_metaphors_flying(self):
        for obj in self.grounded_fliers:
            premise = f"The {obj} is flying."

            yield ( premise, f"The {obj} is moving.", "entailment", 0 )
            yield ( premise, f"The {obj} is moving fast.", "entailment", 1 )
            yield ( premise, f"The {obj} is moving quickly.", "entailment", 2 )

            yield ( premise, f"The {obj} is moving slowly.", "contradiction", 3 )
            yield ( premise, f"The {obj} is not moving.", "contradiction", 4 )


    def gen_orientation_metaphors_down(self):
        for agent in self.agents:
            premise = f"The {agent} is feeling down."
            yield ( premise, f"The {agent} is unhappy.", "entailment", 0 )
            yield ( premise, f"The {agent} is happy.", "contradiction", 1 )


    # TODO: 
    #
    # she's close to resigning -> she's in contact with resigning
    # she's close to 
    #
    # he drank himself out of the promotion -> he was in the promotion
    #
    # He got out out of the chores -> he was in the chores
    # get got out of bed -> he was in bed