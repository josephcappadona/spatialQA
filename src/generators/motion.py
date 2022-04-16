from itertools import product
from .base import BaseGenerator


class MotionGenerator(BaseGenerator):

    reasoning_type = 'motion'

    movement_jerund = {'running', 'sprinting', 'jogging', 'walking', 'jumping', 'swimming'}
    in_motion_phrase = {'moving', 'in motion', 'not stationary'}

    stationary_jerund = {'standing', 'sitting', 'laying down', 'sleeping'}
    not_in_motion_phrase = {'not moving', 'not in motion', 'stationary'}

    neutral_phrase = {'thinking', 'ruminating', 'brainstorming'}

    def gen_motion_one_hop_jerund(self):
        """
            P: AGENT is MOTION
            H1: AGENT is MOTION_PHRASE (E)
            H2: AGENT is NON_MOTON_PHRASE (C)
        """
        for name, motion in product(self.names, self.movement_jerund):
            premise = f"{name} is {motion}."

            for motion_phrase in self.in_motion_phrase:
                hypothesis = f"{name} is {motion_phrase}."
                yield ( premise, hypothesis, self.ENTAILMENT, 0 )

            for non_motion_phrase in self.not_in_motion_phrase:
                hypothesis = f"{name} is {non_motion_phrase}."
                yield ( premise, hypothesis, self.CONTRADICTION, 1 )

            for neutral_phrase in self.neutral_phrase:
                hypothesis = f"{name} is {neutral_phrase}."
                yield ( premise, hypothesis, self.NEUTRAL, 2 )

    def gen_non_motion_one_hop_jerund(self):
        """
        Generates positive textual entailment pairs relating to non-motion of the form:
            P: AGENT is NON_MOTION
            H: AGENT is NON_MOTION_PHRASE

        For example:
            P: John is laying down.
            H: John is stationary.
        """
        for name, not_motion in product(self.names, self.stationary_jerund):

            premise = f"{name} is {not_motion}."

            for not_motion_phrase in self.not_in_motion_phrase:
                hypothesis = f"{name} is {not_motion_phrase}."
                yield ( premise, hypothesis, self.ENTAILMENT, 0 )

            for motion in self.in_motion_phrase:
                hypothesis = f"{name} is {motion}."
                yield ( premise, hypothesis, self.CONTRADICTION, 1 )

            for neutral_phrase in self.neutral_phrase:
                hypothesis = f"{name} is {neutral_phrase}."
                yield ( premise, hypothesis, self.NEUTRAL, 2 )

    def gen_motion_one_hop(self):

        for name, motion_1 in product(self.names, self.in_motion_phrase):

            premise = f"{name} is {motion_1}."

            for motion_2 in self.in_motion_phrase:
                if motion_1 != motion_2:
                    hypothesis = f"{name} is {motion_2}."
                    yield ( premise, hypothesis, self.ENTAILMENT, 0 )

            for not_motion in self.not_in_motion_phrase:
                hypothesis = f"{name} is {not_motion}."
                yield ( premise, hypothesis, self.CONTRADICTION, 1 )

            for neutral_phrase in self.neutral_phrase:
                hypothesis = f"{name} is {neutral_phrase}."
                yield ( premise, hypothesis, self.NEUTRAL, 2 )

    def gen_non_motion_one_hop(self):

        for name, not_motion_1 in product(self.names, self.not_in_motion_phrase):

            premise = f"{name} is {not_motion_1}."

            for not_motion_2 in self.not_in_motion_phrase:
                if not_motion_1 != not_motion_2:
                    hypothesis = f"{name} is {not_motion_2}."
                    yield ( premise, hypothesis, self.ENTAILMENT, 0 )

            for motion in self.in_motion_phrase:
                hypothesis = f"{name} is {motion}."
                yield ( premise, hypothesis, self.CONTRADICTION, 1 )

            for neutral_phrase in self.neutral_phrase:
                hypothesis = f"{name} is {neutral_phrase}."
                yield ( premise, hypothesis, self.NEUTRAL, 2 )
