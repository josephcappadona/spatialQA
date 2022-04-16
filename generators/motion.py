from itertools import product
from .base import BaseGenerator


class MotionGenerator(BaseGenerator):

    reasoning_type = 'motion'

    movement_jerund = ['running', 'sprinting', 'jogging', 'walking', 'jumping', 'swimming']
    stationary_jerund = ['standing', 'sitting', 'laying down', 'sleeping']
    in_motion_phrase = ['moving', 'in motion', 'not stationary']
    not_in_motion_phrase = ['not moving', 'not in motion', 'stationary']
    location_phrase = ['the market', 'school', 'a restaurant']

    action_phrase = ['leave', 'do work later', 'eat lunch', 'be sad']
    state_phrase = ['crazy', 'wild']
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
            yield ( premise, hypothesis, 'entailment' )

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
            yield ( premise, hypothesis, 'contradiction' )

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
            yield ( premise, hypothesis, 'entailment' )

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
            yield ( premise, hypothesis, 'contradiction' )

    def gen_non_motion_neutral(self):
        # TODO: 
        return []

    def gen_go_present_motion_positive(self):
        """
        P: AGENT is going to LOCATION.
        H: AGENT is {MOTION_PHRASE}.
        """

        for agent, location, motion in product(self.names, self.location_phrase, self.in_motion_phrase):
            premise = f"{agent} is going to {location}."
            hypothesis = f"{agent} is {motion}."
            yield ( premise, hypothesis, self.ENTAILMENT)

    def gen_go_present_motion_negative(self):
        """
        P: AGENT is going to LOCATION.
        H: AGENT is NON_MOTION_PHRASE.
        """

        for agent, location, motion in product(self.names, self.location_phrase, self.not_in_motion_phrase):
            premise = f"{agent} is going to {location}."
            hypothesis = f"{agent} is {motion}."
            yield ( premise, hypothesis, self.CONTRADICTION )

    def gen_go_past_motion_positive(self):
        """
        P: AGENT is going to LOCATION.
        H: AGENT is {MOTION_PHRASE}.
        """

        for agent, location, motion in product(self.names, self.location_phrase, self.in_motion_phrase):
            premise = f"{agent} went to {location}."
            hypothesis = f"{agent} was {motion}."
            yield ( premise, hypothesis, self.ENTAILMENT)

    def gen_go_past_motion_negative(self):
        """
        P: AGENT is going to LOCATION.
        H: AGENT is NON_MOTION_PHRASE.
        """

        for agent, location, motion in product(self.names, self.location_phrase, self.not_in_motion_phrase):
            premise = f"{agent} went to {location}."
            hypothesis = f"{agent} was {motion}."
            yield ( premise, hypothesis, self.CONTRADICTION )

    def gen_come_present_motion_positive(self):
        """
        P: AGENT is coming to LOCATION.
        H: AGENT is MOTION_PHRASE.
        """
        for agent, location, motion in product(self.names, self.location_phrase, self.in_motion_phrase):
            premise = f"{agent} is coming to {location}."
            hypothesis = f"{agent} is {motion}."
            yield ( premise, hypothesis, self.ENTAILMENT)

    def gen_come_present_motion_negative(self):
        """
        P: AGENT is coming to LOCATION.
        H: AGENT is MOTION_PHRASE.
        """
        for agent, location, motion in product(self.names, self.location_phrase, self.not_in_motion_phrase):
            premise = f"{agent} is coming to {location}."
            hypothesis = f"{agent} is {motion}."
            yield ( premise, hypothesis, self.CONTRADICTION)

    def gen_come_past_motion_positive(self):
        """
        P: AGENT is coming to LOCATION.
        H: AGENT is MOTION_PHRASE.
        """
        for agent, location, motion in product(self.names, self.location_phrase, self.in_motion_phrase):
            premise = f"{agent} came to {location}."
            hypothesis = f"{agent} was {motion}."
            yield ( premise, hypothesis, self.ENTAILMENT)

    def gen_come_past_motion_negative(self):
        """
        P: AGENT is coming to LOCATION.
        H: AGENT is MOTION_PHRASE.
        """
        for agent, location, motion in product(self.names, self.location_phrase, self.not_in_motion_phrase):
            premise = f"{agent} came to {location}."
            hypothesis = f"{agent} was {motion}."
            yield ( premise, hypothesis, self.CONTRADICTION)

    def gen_metaphor_go_to_action(self):
        for agent, action, motion in product(self.names, self.action_phrase, self.in_motion_phrase):
            premise = f"{agent} is going to {action}."
            hypothesis = f"{agent} is {motion}."
            yield ( premise, hypothesis, self.CONTRADICTION )

    def gen_metaphor_went_state(self):
        for agent, action, motion in product(self.names, self.state_phrase, self.in_motion_phrase):
            premise = f"{agent} went {action}."
            hypothesis = f"{agent} is {motion}."
            yield ( premise, hypothesis, self.CONTRADICTION )