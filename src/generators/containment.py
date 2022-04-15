from itertools import product
from math import hypot
from .base import BaseGenerator
from dataclasses import dataclass
from typing import List


class ContainmentGenerator(BaseGenerator):

    reasoning_type = "containment"

    is_in_phrases = {'is in', 'is inside', 'is contained by'}
    contains_phrases = {'contains', 'is holding'}

    fits_in_phrases = {'can fit in', 'fits in'}
    cannot_fit_in_phrases = {'cannot fit in', 'can\'t fit in', 'doesn\'t fit in'}
    can_contain_phrases = {'can fit', 'fits', 'can hold'}
    cannot_contain_phrases = {'cannot fit', 'cannot hold', 'cannot contain'}

    sm_objects = {'ball', 'block', 'key', 'phone'}
    sm_containers = {'cup', 'bottle', 'small box'}

    med_objects = {'toaster', 'television', 'computer'}
    med_containers = {'cabinet', 'large box', 'suitcase'}

    lg_objects = {'person', 'sofa', 'bookcase', 'table'}
    lg_containers = {'house', 'building', 'warehouse'}

    def gen_one_hop_pos(self):
        for objects, containers \
                in [(self.sm_objects, self.sm_containers),
                    (self.sm_objects, self.med_containers),
                    (self.sm_objects, self.lg_containers),
                    (self.med_objects, self.med_containers),
                    (self.med_objects, self.lg_containers)]:
            
            for obj, cont in product(objects, containers):

                for is_in_phrase in self.is_in_phrases:

                    # The X IS_IN the Y.
                    premise = f"The {obj} {is_in_phrase} the {cont}."

                    for fits_in_phrase in self.fits_in_phrases:
                        # The X FITS_IN the Y.
                        hypothesis = f"The {obj} {fits_in_phrase} the {cont}."
                        yield ( premise, hypothesis, "entailment", 0 )
                    
                    for contains_phrase in self.contains_phrases:
                        # The Y CONTAINS the X.
                        hypothesis = f"The {cont} {contains_phrase} the {obj}."
                        yield ( premise, hypothesis, "entailment", 1 )
                    
                    for can_contain_phrase in self.can_contain_phrases:
                        # The Y CAN_CONTAIN the X.
                        hypothesis = f"The {cont} {can_contain_phrase} the {obj}."
                        yield ( premise, hypothesis, "entailment", 2 )
    
    def gen_one_hop_neg(self):
        for objects, containers \
                in [(self.med_objects, self.sm_containers),
                    (self.lg_objects, self.sm_containers),
                    (self.lg_objects, self.med_containers)]:

            for obj, cont in product(objects, containers):

                # The X CANNOT_FIT_IN the Y.
                for cannot_fit_in_phrase in self.cannot_fit_in_phrases:
                    premise = f"The {obj} {cannot_fit_in_phrase} the {cont}."

                    # The Y CAN_CONTAIN the X.
                    for can_contain_phrase in self.can_contain_phrases:
                        hypothesis = f"The {cont} {can_contain_phrase} the {obj}."
                        yield ( premise, hypothesis, "contradiction", 0 )
                    
                    # The Y CANNOT_CONTAIN the X.
                    for cannot_contain_phrase in self.cannot_contain_phrases:
                        hypothesis = f"The {cont} {cannot_contain_phrase} the {obj}."
                        yield ( premise, hypothesis, "entailment", 1 )
    
    def gen_two_hop(self):

        for smaller_objs, smaller_conts, larger_conts \
                in [(self.sm_objects, self.sm_containers, self.med_containers),
                    (self.med_objects, self.med_containers, self.lg_containers),
                    (self.sm_objects, self.sm_containers, self.lg_containers),
                    (self.sm_objects, self.med_containers, self.lg_containers)]:

            for sm_obj, sm_cont, med_cont \
                    in product(smaller_objs, smaller_conts, larger_conts):
                
                for is_in_1, is_in_2 \
                        in product(self.is_in_phrases, self.is_in_phrases):
                
                    # 0. The X IS_IN the Y. The Y IS_IN the Z.
                    premise = f"The {sm_obj} {is_in_1} the {sm_cont}. The {sm_cont} {is_in_2} the {med_cont}."

                    for is_in_phrase_3 in self.is_in_phrases:
                        # 0. The X is in the Z. True
                        hypothesis = f"The {sm_obj} {is_in_phrase_3} the {med_cont}."
                        yield ( premise, hypothesis, "entailment", 0 )
                    
                    for contains_phrase in self.contains_phrases:
                        # 1. The Y CONTAINS the X. True
                        hypothesis = f"The {sm_cont} {contains_phrase} the {sm_obj}."
                        yield ( premise, hypothesis, "entailment", 1 )

                        # 2. The Z CONTAINS the X. True
                        hypothesis = f"The {med_cont} {contains_phrase} the {sm_obj}."
                        yield ( premise, hypothesis, "entailment", 2 )

                        # 3. The Z CONTAINS the Y. True
                        hypothesis = f"The {med_cont} {contains_phrase} the {sm_cont}."
                        yield ( premise, hypothesis, "entailment", 3 )

                    for fits_in_phrase in self.fits_in_phrases:
                        # 4. The X CAN_FIT_IN the Z. True
                        hypothesis = f"The {sm_obj} {fits_in_phrase} the {med_cont}."
                        yield ( premise, hypothesis, "entailment", 4 )

                    for fits_in_phrase in self.fits_in_phrases:
                        # 5. The Z CAN_FIT_IN the Y. False
                        hypothesis = f"The {med_cont} {fits_in_phrase} the {sm_cont}."
                        yield ( premise, hypothesis, "contradiction", 5 )
                    
                    for contains_phrase in self.contains_phrases:
                        # 6. The Y CONTAINS the Z. False
                        hypothesis = f"The {sm_cont} {contains_phrase} the {med_cont}."
                        yield ( premise, hypothesis, "contradiction", 6 )
                
                for fits_in_1, fits_in_2 \
                        in product(self.fits_in_phrases, self.fits_in_phrases):
                    
                    # The X FITS_IN the Y. The Y FITS_IN the Z.
                    premise = f"The {sm_obj} {fits_in_1} the {sm_cont}. The {sm_cont} {fits_in_2} the {med_cont}."

                    for fits_in_phrase_3 in self.fits_in_phrases:
                        # The X FITS_IN the Z. True
                        hypothesis = f"The {sm_obj} {fits_in_phrase_3} the {med_cont}."
                        yield ( premise, hypothesis, "entailment", 7 )
                    
                    for is_in_phrase in self.is_in_phrases:
                        # The X IS_IN the Y. Neither
                        hypothesis = f"The {sm_obj} {is_in_phrase} the {sm_cont}."
                        yield ( premise, hypothesis, "neutral", 8 )

                        # The X IS_IN the Z. Neither
                        hypothesis = f"The {sm_obj} {is_in_phrase} the {med_cont}."
                        yield ( premise, hypothesis, "neutral", 9 )

                        # The Y IS_IN the Z. Neither
                        hypothesis = f"The {sm_cont} {is_in_phrase} the {med_cont}."
                        yield ( premise, hypothesis, "neutral", 10 )
                    
                    for contains_phrase in self.contains_phrases:
                        # The Y CONTAINS the X. Neither
                        hypothesis = f"The {sm_cont} {contains_phrase} the {sm_obj}."
                        yield ( premise, hypothesis, "neutral", 11 )

                        # The Z CONTAINS the X. Neither
                        hypothesis = f"The {med_cont} {contains_phrase} the {sm_obj}."
                        yield ( premise, hypothesis, "neutral", 12 )

                        # The Z CONTAINS the Y. Neither
                        hypothesis = f"The {med_cont} {contains_phrase} the {sm_cont}."
                        yield ( premise, hypothesis, "neutral", 13 )



