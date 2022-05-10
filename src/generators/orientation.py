from itertools import product
from .base import BaseGenerator


class OrientationGenerator(BaseGenerator):

    reasoning_type = 'orientation'

    objects = {'block', 'book', 'box', 'cup', 'crate'}
    locations = {'theater', 'school', 'park'}
    cardinals = {'north', 'south', 'east', 'west'} #'northeast', 'northwest', 'southeast', 'southwest'}
    above_phrases = {'above', 'on top of'}
    below_phrases = {'below', 'underneath'}
    left_phrases = {'left of', 'to the left of', 'on the left side of'}
    right_phrases = {'right of', 'to the right of', 'on the right side of'}
    non_above_phrases = below_phrases.union(left_phrases).union(right_phrases)
    non_below_phrases = above_phrases.union(left_phrases).union(right_phrases)
    non_left_phrases = right_phrases.union(above_phrases).union(below_phrases)
    non_right_phrases = left_phrases.union(above_phrases).union(below_phrases)

    def gen_cardinals_one_hop(self):
        """
        Generates positive textual entailment pairs relating to orientation of the form:
            P: The LOC_1 is CARDINAL_DIR of the LOC_2.
            H: The LOC_2 is OPPOSITE(CARD) of the LOC_1.

        For example:
            P: The post office is north of the library.
            H: The library is south of the post office.
        """
        for loc_1, cardinal_1, loc_2 in product(self.locations, self.cardinals, self.locations):
            if loc_1 != loc_2:
                premise = f"The {loc_1} is {cardinal_1} of the {loc_2}."

                hypothesis = f"The {loc_2} is {self.opposite_cardinal(cardinal_1)} of the {loc_1}."
                yield ( premise, hypothesis, self.ENTAILMENT, 0 )

                for cardinal_2 in self.non_opposite_cardinals(cardinal_1):
                    yield ( premise, f"The {loc_2} is {cardinal_2} of the {loc_1}.", self.CONTRADICTION, 1 )

    def gen_cardinals_neutral(self):
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
                        yield ( premise, f"The {loc_1} is {card_2} of the {loc_3}.", self.NEUTRAL, 0 )

    def gen_left_right_one_hop(self):
        for dir_rels in [self.left_phrases, self.right_phrases]:
            for obj_1, obj_2, dir in product(self.objects, self.objects, dir_rels):
                if obj_1 != obj_2:
                    premise = f"The {obj_1} is {dir} the {obj_2}."

                    for rev_dir in self.opposite_directions(dir):
                        yield ( premise, f"The {obj_2} is {rev_dir} the {obj_1}.", self.ENTAILMENT, 0 )
                    
                    for non_rev_dir in self.non_opposite_directions(dir):
                        yield ( premise, f"The {obj_2} is {non_rev_dir} the {obj_1}.", self.CONTRADICTION, 1 )

    def gen_above_below_one_hop(self):
        for dir_rels in [self.above_phrases, self.below_phrases]:
            for obj_1, obj_2, dir in product(self.objects, self.objects, dir_rels):
                if obj_1 != obj_2:
                    premise = f"The {obj_1} is {dir} the {obj_2}."

                    for rev_dir in self.opposite_directions(dir):
                        yield ( premise, f"The {obj_2} is {rev_dir} the {obj_1}.", self.ENTAILMENT, 0 )
                    
                    for non_rev_dir in self.non_opposite_directions(dir):
                        yield ( premise, f"The {obj_2} is {non_rev_dir} the {obj_1}.", self.CONTRADICTION, 1 )
    

    def opposite_directions(self, dir):
        if dir in self.above_phrases:
            return self.below_phrases
        elif dir in self.below_phrases:
            return self.above_phrases
        elif dir in self.left_phrases:
            return self.right_phrases
        elif dir in self.right_phrases:
            return self.left_phrases
        else:
            return set()
    
    def non_opposite_directions(self, dir):
        if dir in self.above_phrases:
            return self.non_below_phrases
        elif dir in self.below_phrases:
            return self.non_above_phrases
        elif dir in self.left_phrases:
            return self.non_right_phrases
        elif dir in self.right_phrases:
            return self.non_left_phrases
        else:
            return set()

    def opposite_cardinal(self, dir):
        if dir == 'north': return 'south'
        elif dir == 'south': return 'north'
        elif dir == 'east': return 'west'
        elif dir == 'west': return 'east'
        elif dir == 'northeast': return 'southwest'
        elif dir == 'southeast': return 'northwest'
        elif dir == 'northwest': return 'southeast'
        elif dir == 'southwest': return 'northeast'

    def non_opposite_cardinals(self, cardinal):
        opposite_cardinal = self.opposite_cardinal(cardinal)
        return self.cardinals.difference({ opposite_cardinal })