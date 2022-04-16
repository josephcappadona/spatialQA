from itertools import product
from .base import BaseGenerator


class MetaphorGenerator(BaseGenerator):

    reasoning_type = 'metaphor'

    machines = {'car', 'robot', 'plane'}
    in_motion_phrases = {'motion', 'flight', 'action'}

    agents = {'girl', 'woman', 'boy', 'man'}
    tasks = {'washing the windows', 'cleaning the table', 'sweeping the floor'}
    tasks_past = {'washed the windows', 'cleaned the table', 'sweeped the floor'}
    emotions = {'happiness', 'anger', 'sadness'}
    materials = {'cotton', 'wool', 'polyester'}
    material_objects = {'shirt', 'sweater', 'jacket'}
    states = {'disguise', 'shape', 'a bad mood', 'a good mood', 'love'}

    objects = {'ball', 'dax', 'widget', 'card'}
    grounded_fliers = {'car', 'dog', 'horse', 'cat'}
    jump_dests = {'to the next section', 'to the end of the movie', 'to conclusions'}
    events_to_skip = {'movie', 'class', 'show', 'introduction'}
    agreements = {'agreement', 'a deal', 'a compromise'}

    events_to_put_behind = {'fiasco', 'tragedy', 'event'}
    events_to_look_forward = {'party', 'concert', 'presentation'}

    achievings = {'finishing', 'completing', 'being done with'}
    goals = {'goal', 'task', 'job'}


    def gen_containment_metaphors(self):
        for obj, motion in product(self.objects, self.in_motion_phrases):
            premise = f"The {obj} is in {motion}."
            hypothesis = f"The {obj} is inside {motion}."
            yield ( premise, hypothesis, "contradiction,neutral", 0 )

        for agent, task in product(self.agents, self.tasks):
            premise = f"The {agent} got out of {task}."
            hypothesis = f"The {agent} was inside {task}."
            yield ( premise, hypothesis, "contradiction,neutral", 1 )
        
        for agent, emotion in product(self.agents, self.emotions):
            premise = f"The {agent} could not contain their {emotion}."
            hypothesis = f"The {emotion} was located inside {agent}."
            yield ( premise, hypothesis, "contradiction,neutral", 2 )
        
        for material, material_object in product(self.materials, self.material_objects):
            premise = f"There is {material} in that {material_object}."
            hypothesis = f"There is {material} located within {material_object}."
            yield ( premise, hypothesis, "contradiction,neutral", 3 )
        
        for name, state in product(self.names, self.states):
            premise = f"{name} is in {state}."
            hypothesis = f"{name} is physically contained in {state}."
            yield ( premise, hypothesis, "contradiction,neutral", 4 )

        for name, task in product(self.names, self.tasks):
            premise = f"{name} put a lot of energy into {task}."
            hypothesis_1 = f"There is now energy inside {task}."
            hypothesis_2 = f"Energy is now inside {task}."
            yield ( premise, hypothesis_1, "contradiction,neutral", 5 )
            yield ( premise, hypothesis_2, "contradiction,neutral", 6 )
        
        for name, task_1, task_2 in product(self.names, self.tasks, self.tasks_past):
            if task_1 != task_2:
                premise = f"Outside of {task_1}, {name} also {task_2}."
                hypothesis = f"{name} is located outside of {task_1}."
                yield ( premise, hypothesis, "contradiction,neutral", 7 )
        
        for name, emotion in product(self.names, self.emotions):
            premise = f"{name} is filled with {emotion}."
            hypothesis = f"Happiness is located inside {name}."
            yield ( premise, hypothesis, "contradiction,neutral", 8 )


    def gen_motion_metaphors(self):
        for name, event in product(self.names, self.events_to_skip):
            premise = f"{name} is skipping the {event}."

            yield ( premise, f"{name} is in motion.", "contradiction", 0 )
            yield ( premise, f"{name} is moving.", "contradiction", 1 )

        for name, jump_dest in product(self.names, self.jump_dests):
            premise = f"{name} is jumping {jump_dest}."

            yield ( premise, f"{name} is in motion.", "contradiction", 2 )
            yield ( premise, f"{name} is moving.", "contradiction", 3 )

        for obj in self.grounded_fliers:
            premise = f"The {obj} is flying."
            yield ( premise, f"The {obj} is moving.", "entailment", 4 )
            yield ( premise, f"The {obj} is airborne.", "contradiction,neutral", 5 )

        for name_1, name_2, agreement in product(self.names, self.names, self.agreements):
            if name_1 != name_2:
                premise = f"{name_1} and {name_2} are moving toward {agreement}."
                hypothesis = f"{name_1} and {name_2} are in motion."
                yield ( premise, hypothesis, "contradiction,neutral", 6 )


    def gen_orientation_metaphors(self):
        for agent in self.agents:
            premise = f"The {agent} is feeling down."
            yield ( premise, f"The {agent} is unhappy.", "entailment", 0 )
            yield ( premise, f"The {agent} is happy.", "contradiction", 1 )
        
        for agent, event in product(self.agents, self.events_to_put_behind):
            premise = f"The {agent} put the {event} behind them."
            hypothesis = f"The {agent} is in front of the {event}."
            yield ( premise, hypothesis, "contradiction,neutral", 2 )

        for agent, event in product(self.agents, self.events_to_look_forward):
            premise = f"The {agent} is looking forward to the {event}."
            hypothesis = f"The {agent} is behind the {event}."
            yield ( premise, hypothesis, "contradiction,neutral", 3 )


    def gen_distance_metaphors(self):
        for name, achieving, goal in product(self.names, self.achievings, self.goals):
            premise = f"{name} is close to {achieving} the {goal}."
            hypothesis_1 = f"{name} is in close proximity to the {goal}."
            hypothesis_2 = f"{name} and the {goal} are in close proximity."
            hypothesis_3 = f"{name} and the {goal} are near one another."
            yield ( premise, hypothesis_1, "contradiction,neutral", 0 )
            yield ( premise, hypothesis_2, "contradiction,neutral", 1 )
            yield ( premise, hypothesis_3, "contradiction,neutral", 2 )

        for name, achieving, goal in product(self.names, self.achievings, self.goals):
            premise = f"{name} is far from {achieving} the {goal}."
            hypothesis_1 = f"{name} and the {goal} are not in close proximity."
            hypothesis_2 = f"{name} and the {goal} are far away from one another."
            yield ( premise, hypothesis_1, "contradiction,neutral", 3 )
            yield ( premise, hypothesis_2, "contradiction,neutral", 4 )


    # TODO: 
    #
    # she's close to resigning -> she's in contact with resigning
    # she's close to 
    #
    # he drank himself out of the promotion -> he was in the promotion
    #
    # He got out out of the chores -> he was in the chores
    # get got out of bed -> he was in bed