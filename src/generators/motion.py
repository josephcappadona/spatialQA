from itertools import product
from .base import BaseGenerator


class MotionGenerator(BaseGenerator):

    reasoning_type = 'motion'

    movement_jerund = ['running', 'sprinting', 'jogging', 'walking', 'jumping', 'swimming']
    stationary_jerund = ['standing', 'sitting', 'laying down', 'sleeping']
    in_motion_phrase = ['moving', 'in motion', 'not stationary']
    not_in_motion_phrase = ['not moving', 'not in motion', 'stationary']

    def gen_motion_positive(self):
        """
        Generates positive textual entailment pairs relating to motion of the form:
            P: AGENT is MOTION
            H: AGENT is MOTION_PHRASE

        For example:
            P: John is running.
            H: John is in motion.
        """
        for agent, motion, motion_phrase in product(self.names, self.movement_jerund, self.in_motion_phrase):
            premise = f"{agent} is {motion}."
            hypothesis = f"{agent} is {motion_phrase}."
            yield ( premise, hypothesis, "entailment", 0 )

    def gen_motion_negative(self):
        """
        Generates negative textual entailment pairs relating to motion of the form:
            P: AGENT is MOTION
            H: AGENT is NON_MOTION_PHRASE

        For example:
            P: John is running.
            H: John is stationary.
        """
        for agent, motion, non_motion_phrase \
            in product(self.names, self.movement_jerund, self.not_in_motion_phrase):

            premise = f"{agent} is {motion}."
            hypothesis = f"{agent} is {non_motion_phrase}."
            yield ( premise, hypothesis, "contradiction", 0 )

    def gen_motion_neutral(self):
        # TODO: 
        return []

    def gen_non_motion_positive(self):
        """
        Generates positive textual entailment pairs relating to non-motion of the form:
            P: AGENT is NON_MOTION
            H: AGENT is NON_MOTION_PHRASE

        For example:
            P: John is laying down.
            H: John is stationary.
        """
        for agent, not_motion, not_motion_phrase in product(self.names, self.stationary_jerund, self.not_in_motion_phrase):
            premise = f"{agent} is {not_motion}."
            hypothesis = f"{agent} is {not_motion_phrase}."
            yield ( premise, hypothesis, "entailment", 0 )

    def gen_non_motion_negative(self):
        """
        Generates negative textual entailment pairs relating to non-motion of the form:
            P: AGENT is NON_MOTION
            H: AGENT is MOTION_PHRASE

        For example:
            P: John is laying down.
            H: John is in motion.
        """
        for agent, not_motion, motion in product(self.names, self.stationary_jerund, self.in_motion_phrase):
            premise = f"{agent} is {not_motion}."
            hypothesis = f"{agent} is {motion}."
            yield ( premise, hypothesis, "contradiction", 0 )

    def gen_non_motion_neutral(self):
        # TODO: 
        return []
