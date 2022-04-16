from itertools import product
from .base import BaseGenerator


class ContainmentGenerator(BaseGenerator):

    reasoning_type = 'containment'

    place_in_verb = {'placed', 'deposited', 'left', 'arranged', 'dumped', 'parked'}
    place_into_verb = {'moved', 'placed', 'deposited', 'transported'}
    metaphor_into_verb = {'forced', 'spurred', 'moved'}

    agent = {'the employee', 'a student', 'the child'}.union(BaseGenerator.names)
    action = {'silence', 'movement', 'crying', 'action'}

    contain_rel = {'on the inside of', 'within', 'contained by'}
    is_contained_rel = {'on the outside of', 'not contained by'}

    object = {'the apples', 'puppies', 'some basketballs'}
    container = {'the bucket', 'a bag', 'the car', 'a car'}
    emotion = {'anger', 'sadness', 'regret', 'joy'}

    def gen_movement_into_container(self):
        """
        P: The apples are placed in buckets.
        H1: The apples are located inside buckets.
        H2: The apples are located outside buckets.
        H3: The apples are located near buckets.
        """
        for entity, movement_rel, container in product(self.object, self.place_into_verb, self.container):
            premise = f"{entity} are {movement_rel} into {container}."
            for contain_rel in self.contain_rel:
                yield ( premise, f"{entity} are {contain_rel} {container}.", self.ENTAILMENT )
                yield ( premise, f"{container} is {contain_rel} {entity}.", self.CONTRADICTION )
            for contain_rel in self.is_contained_rel:
                yield ( premise, f"{entity} are {contain_rel} {container}.", self.CONTRADICTION )
                yield ( premise, f"{container} is {contain_rel} {entity}.", self.CONTRADICTION )

    def gen_metaphor_action_into_movement(self):
        """
        P: The workers were spurred into movement.
        H1: The workers were inside movement.
        H2: The workers were outside movement.
        {agent} got/were {force_verb} into {action}.	{agent} was inside/outside {action}.
        """
        for agent, movement_rel, action, contain_rel in product(self.agent, self.metaphor_into_verb, self.action, self.contain_rel):
            premise = f"{agent} is {movement_rel} into {action}."
            yield ( premise, f"{agent} is {contain_rel} {action}.", self.CONTRADICTION )
            yield ( premise, f"{action} is {contain_rel} {agent}.", self.CONTRADICTION )

    def metaphor_action_in_motion(self):
        """
        P:
        {machine} is in motion/flight/action.	{machine} is inside {motion}.
        """

        """
        outside of {task}, {agent} also {did task}.	{agent} is located outside of {task}.
        """

        """
        {agent} {neg_action} {self} out of {consideration}.	{agent} was located inside {consideration}.
        """

        """
        {agent} put a lot of {effort} into {task}.	{effort} is inside {task}. {agent} put {effort} inside {task}. {agent} put {effort} inside {task}.
        """

    def gen_motion_out_of(self):
        """
        P: John will get out of the car.
        H1: John is on the inside of the car.
        H2: The car is on the inside of John.
        H3: John is on the outside of the car.
        H2: The car is on the outside of John.
        """

        containing_body = {'car', 'elevator', 'building'}
        for agent, containing_body in product(self.agent, containing_body):
            premise = f"{agent} will get out of the {containing_body}."
            for contain_rel in self.contain_rel:
                yield (premise, f"{agent} is {contain_rel} the {containing_body}.", self.ENTAILMENT)
                yield (premise, f"The {containing_body} is {contain_rel} {agent}.", self.CONTRADICTION)
            for contain_rel in self.is_contained_rel:
                yield (premise, f"{agent} is {contain_rel} the {containing_body}.", self.CONTRADICTION)
                yield (premise, f"The {containing_body} is {contain_rel} the {agent}.", self.ENTAILMENT)


    def gen_metaphor_motion_out_of(self):
        """
        P: Tom got out of washing the windows.
        H1: Tom is in washing the windows.
        H2: Washing the windows is in Tom.
        H3: Tom is outside of washing the windows.
        H4: Washing the windows is outside of Tom.
        {agent} got out of {task}.	{agent} was located inside {task}.
        """
        task = {'washing the windows', 'doing chores', 'mowing the lawn'}
        for agent, task in product(self.agent, task):
            premise = f"{agent} got out of {task}."
            for contain_rel in self.contain_rel:
                yield (premise, f"{agent} is {contain_rel} the {task}.", self.CONTRADICTION)
                yield (premise, f"The {task} is {contain_rel} {agent}.", self.CONTRADICTION)
            for contain_rel in self.is_contained_rel:
                yield (premise, f"{agent} is {contain_rel} the {task}.", self.CONTRADICTION)
                yield (premise, f"The {task} is {contain_rel} the {agent}.", self.CONTRADICTION)

    def gen_contain_objects(self):
        """
        P: The basket contains apples.
        H1: The apples are inside the basket.
        H2: The basket is inside the apples.
        """
        for object, container, contain_rel in product(self.object, self.container, self.contain_rel):
            premise = f"{container} contains {object}."
            yield ( premise, f"{object} are {contain_rel} {container}.", self.ENTAILMENT)
            yield ( premise, f"{container} is {contain_rel} {object}.", self.CONTRADICTION)

    def gen_metaphor_contain_emotions(self):
        """
        P: Tom could not contain his joy.
        {agent} could not contain {self's} {emotion}.	{agent} is a container.
        """
        for agent, emotion, contain_rel in product(self.agent, self.emotion, self.contain_rel):
            premise = f"{agent} could not contain their {emotion}."
            yield ( premise, f"{emotion} is physically {contain_rel} {agent}.", self.CONTRADICTION)
            yield ( premise, f"{agent} is physically {contain_rel} {emotion}.", self.CONTRADICTION)

    def gen_filled_with(self):
        """
        P: The basket is filled with apples.
        H1: Apples are on the inside of the basket.
        H2: The basket is on the inside of apples.
        """
        for container, object, contain_rel in product(self.container, self.object, self.contain_rel):
            premise = f"{container} is filled with {object}."
            yield ( premise, f"{object} are physically {contain_rel} {container}.", self.ENTAILMENT)
            yield ( premise, f"{container} is physically {contain_rel} {object}.", self.CONTRADICTION)

    def gen_metaphor_filled_with(self):
        """
        {agent} was filled with {emotion}.	{agent} is a container.
        """
        for agent, emotion, contain_rel in product(self.agent, self.emotion, self.contain_rel):
            premise = f"{agent} is filled with {emotion}."
            yield ( premise, f"{emotion} is physically {contain_rel} {agent}.", self.CONTRADICTION)
            yield ( premise, f"{agent} is physically {contain_rel} {emotion}.", self.CONTRADICTION)

    def gen_object_in_container(self):
        """
        P: The apples are in the basket.
        H1: The apples are inside the basket.
        H2: The basket is inside the apples.
        """
        for object, container, contain_rel in product(self.object, self.container, self.contain_rel):
            premise = f"{container} is in {object}."
            yield ( premise, f"{object} are {contain_rel} {container}.", self.ENTAILMENT)
            yield ( premise, f"{container} is {contain_rel} {object}.", self.CONTRADICTION)

    def gen_metaphor_entity_in_state(self):
        """
        {entity} is in {state}.	{entity} is physically contained in {state}.
        P: The student are in trouble.
        H1: Trouble is inside the student.
        H2: The student is inside trouble.
        """
        state = {'disguise', 'shape', 'a bind', 'trouble'}
        for agent, state, contain_rel in product(self.agent, state, self.contain_rel):
            premise = f"{agent} is in {state}."
            yield ( premise, f"{agent} is physically {contain_rel} {state}.", self.CONTRADICTION)
            yield ( premise, f"{state} is physically {contain_rel} {agent}.", self.CONTRADICTION)




        """
        {agent} could not get {information} out of {info_source}.	{information} was in {info_source}.
        """


        """
        {subtype} falls into the category of {supertype}.	{subtype} is within/contained in {supertype}.
        """


        """
        The {timeline} has been full (of {events}).	{timeline} is a container.
        """

        """
        {timeline} is packed with {events}.	{events} are located within {timeline}.
        """