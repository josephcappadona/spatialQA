from itertools import product
from .base import BaseGenerator


class OrientationGenerator(BaseGenerator):

    reasoning_type = 'orientation'

    locations = {'theater', 'school', 'park'}
    cardinals = {'north', 'south', 'east', 'west', 'northeast', 'southeast', 'northwest', 'southwest'}
    #directions = {'above', 'below'}

    def gen_orientation_cardinals_positive(self):
        """
        Generates positive textual entailment pairs relating to orientation of the form:
            P: The LOC_1 is CARDINAL_DIR of the LOC_2.
            H: The LOC_2 is OPPOSITE(CARD) of the LOC_1.

        For example:
            P: The post office is north of the library.
            H: The library is south of the post office.
        """
        for loc_1, cardinal, loc_2 in product(self.locations, self.cardinals, self.locations):
            if loc_1 != loc_2:
                premise = f"The {loc_1} is {cardinal} of the {loc_2}."
                hypothesis = f"The {loc_2} is {self.opposite_cardinal(cardinal)} of the {loc_1}."
                yield ( premise, hypothesis, 'entailment' )

    def gen_orientation_cardinals_negative(self):
        """
        Generates negative textual entailment pairs relating to orientation of the form:
            P: The LOC_1 is CARDINAL_DIR of the LOC_2.
            H: The LOC_2 is NON_OPPOSITE(CARD) of the LOC_1.

        For example:
            P: The post office is north of the library.
            H: The library is east of the post office.
        """
        for loc_1, cardinal_1, loc_2 in product(self.locations, self.cardinals, self.locations):
            if loc_1 != loc_2:
                premise = f"The {loc_1} is {cardinal_1} of the {loc_2}."
                for cardinal_2 in self.non_opposite_cardinals(cardinal_1):
                    yield ( premise, f"The {loc_2} is {cardinal_2} of the {loc_1}.", 'contradiction' )

    def gen_orientation_cardinals_neutral(self):
        """
        Generates neutral textual entailment pairs relating to orientation of the form:
            P: The LOC_1 is CARDINAL_DIR_1 of the LOC_2.
            H: The LOC_1 is CARDINAL_DIR_2 of the LOC_3.

        For example:
            P: The post office is north of the library.
            H: The post office is east of the school.
        """
        for loc_1, card_1, loc_2 in product(self.locations, self.cardinals, self.locations):
            if loc_1 != loc_2:
                premise = f"The {loc_1} is {card_1} of the {loc_2}."
                for loc_3 in self.locations.difference({ loc_1, loc_2}):
                    for card_2 in self.cardinals:
                        yield ( premise, f"The {loc_1} is {card_2} of the {loc_3}.", 'neutral' )
        
        # TODO: same as above but for objects left/right on a table


    def opposite_cardinal(self, dir):
        if dir == 'north': return 'south'
        if dir == 'south': return 'north'
        if dir == 'east': return 'west'
        if dir == 'west': return 'east'
        if dir == 'northeast': return 'southwest'
        if dir == 'northwest': return 'southeast'
        if dir == 'southeast': return 'northwest'
        if dir == 'southwest': return 'northeast'

    def non_opposite_cardinals(self, cardinal):
        opposite_cardinal = self.opposite_cardinal(cardinal)
        return self.cardinals.difference({ opposite_cardinal })