from itertools import product
from math import hypot
from .base import BaseGenerator
from dataclasses import dataclass
from typing import List


class ContainmentGenerator(BaseGenerator):

    reasoning_type = "containment"

    agents = {'a student', 'the child'}.union(BaseGenerator.names)

    in_rels = {'in', 'inside', 'contained by'}
    contains_phrases = {'contains', 'is holding'}
    outside_rels = {'on the outside of', 'not contained by'}

    fits_in_phrases = {'can fit in', 'fits in'}
    cannot_fit_in_phrases = {'cannot fit in', 'can\'t fit in'}
    can_contain_phrases = {'can fit', 'fits'}
    cannot_contain_phrases = {'cannot fit', 'cannot hold'}

    sm_objects = {'block', 'pen', 'pencil'}
    sm_containers = {'cup', 'small box'}

    med_objects = {'toaster', 'television', 'computer'}
    med_containers = {'cabinet', 'suitcase'}

    lg_objects = {'person', 'sofa', 'bookcase'}
    lg_containers = {'house', 'building', 'warehouse'}

    plural_objects = {'the apples', 'some basketballs'}
    containers = {'the bucket', 'a bag', 'the car'}

    place_into_verb = {'moved', 'placed', 'transported'}

    def gen_one_hop_pos(self):
        for objects, containers \
                in [(self.sm_objects, self.sm_containers),
                    (self.sm_objects, self.med_containers),
                    (self.sm_objects, self.lg_containers),
                    (self.med_objects, self.med_containers),
                    (self.med_objects, self.lg_containers)]:
            
            for obj, cont in product(objects, containers):

                for in_phrase in self.in_rels:

                    # The X is IN the Y.
                    premise = f"The {obj} is {in_phrase} the {cont}."

                    for fits_in_phrase in self.fits_in_phrases:
                        # The X FITS_IN the Y.
                        hypothesis = f"The {obj} {fits_in_phrase} the {cont}."
                        yield ( premise, hypothesis, self.ENTAILMENT, 0 )
                    
                    for contains_phrase in self.contains_phrases:
                        # The Y CONTAINS the X.
                        hypothesis = f"The {cont} {contains_phrase} the {obj}."
                        yield ( premise, hypothesis, self.ENTAILMENT, 1 )
                    
                    for can_contain_phrase in self.can_contain_phrases:
                        # The Y CAN_CONTAIN the X.
                        hypothesis = f"The {cont} {can_contain_phrase} the {obj}."
                        yield ( premise, hypothesis, self.ENTAILMENT, 2 )
    
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
                        yield ( premise, hypothesis, self.CONTRADICTION, 0 )
                    
                    # The Y CANNOT_CONTAIN the X.
                    for cannot_contain_phrase in self.cannot_contain_phrases:
                        hypothesis = f"The {cont} {cannot_contain_phrase} the {obj}."
                        yield ( premise, hypothesis, self.ENTAILMENT, 1 )
    
    def gen_two_hop_is_in(self):

        for smaller_objs, smaller_conts, larger_conts \
                in [(self.sm_objects, self.sm_containers, self.med_containers),
                    (self.med_objects, self.med_containers, self.lg_containers),
                    (self.sm_objects, self.sm_containers, self.lg_containers),
                    (self.sm_objects, self.med_containers, self.lg_containers)]:

            for sm_obj, sm_cont, med_cont \
                    in product(smaller_objs, smaller_conts, larger_conts):
                
                for in_rel_1, in_rel_2 \
                        in product(self.in_rels, self.in_rels):
                
                    # 0. The X is IN the Y. The Y is IN the Z.
                    premise = f"The {sm_obj} is {in_rel_1} the {sm_cont}. The {sm_cont} is {in_rel_2} the {med_cont}."

                    for in_rel_3 in self.in_rels:
                        # 0. The X is in the Z. True
                        hypothesis = f"The {sm_obj} is {in_rel_3} the {med_cont}."
                        yield ( premise, hypothesis, self.ENTAILMENT, 0 )
                    
                    for contains_phrase in self.contains_phrases:
                        # 1. The Y CONTAINS the X. True
                        hypothesis = f"The {sm_cont} {contains_phrase} the {sm_obj}."
                        yield ( premise, hypothesis, self.ENTAILMENT, 1 )

                        # 2. The Z CONTAINS the X. True
                        hypothesis = f"The {med_cont} {contains_phrase} the {sm_obj}."
                        yield ( premise, hypothesis, self.ENTAILMENT, 2 )

                        # 3. The Z CONTAINS the Y. True
                        hypothesis = f"The {med_cont} {contains_phrase} the {sm_cont}."
                        yield ( premise, hypothesis, self.ENTAILMENT, 3 )

                    for fits_in_phrase in self.fits_in_phrases:
                        # 4. The X CAN_FIT_IN the Z. True
                        hypothesis = f"The {sm_obj} {fits_in_phrase} the {med_cont}."
                        yield ( premise, hypothesis, self.ENTAILMENT, 4 )

                    for fits_in_phrase in self.fits_in_phrases:
                        # 5. The Z CAN_FIT_IN the Y. False
                        hypothesis = f"The {med_cont} {fits_in_phrase} the {sm_cont}."
                        yield ( premise, hypothesis, self.CONTRADICTION, 5 )
                    
                    for contains_phrase in self.contains_phrases:
                        # 6. The Y CONTAINS the Z. False
                        hypothesis = f"The {sm_cont} {contains_phrase} the {med_cont}."
                        yield ( premise, hypothesis, self.CONTRADICTION, 6 )
    
    def gen_two_hop_fits_in(self):

        for smaller_objs, smaller_conts, larger_conts \
                in [(self.sm_objects, self.sm_containers, self.med_containers),
                    (self.med_objects, self.med_containers, self.lg_containers),
                    (self.sm_objects, self.sm_containers, self.lg_containers),
                    (self.sm_objects, self.med_containers, self.lg_containers)]:

            for sm_obj, sm_cont, med_cont \
                    in product(smaller_objs, smaller_conts, larger_conts):
                
                for fits_in_1, fits_in_2 \
                        in product(self.fits_in_phrases, self.fits_in_phrases):
                    
                    # The X FITS_IN the Y. The Y FITS_IN the Z.
                    premise = f"The {sm_obj} {fits_in_1} the {sm_cont}. The {sm_cont} {fits_in_2} the {med_cont}."

                    for fits_in_phrase_3 in self.fits_in_phrases:
                        # The X FITS_IN the Z. True
                        hypothesis = f"The {sm_obj} {fits_in_phrase_3} the {med_cont}."
                        yield ( premise, hypothesis, self.ENTAILMENT, 0 )
                    
                    for in_rel in self.in_rels:
                        # The X is IN the Y. Neither
                        hypothesis = f"The {sm_obj} is {in_rel} the {sm_cont}."
                        yield ( premise, hypothesis, self.NEUTRAL, 1 )

                        # The X is IN the Z. Neither
                        hypothesis = f"The {sm_obj} is {in_rel} the {med_cont}."
                        yield ( premise, hypothesis, self.NEUTRAL, 2 )

                        # The Y is IN the Z. Neither
                        hypothesis = f"The {sm_cont} is {in_rel} the {med_cont}."
                        yield ( premise, hypothesis, self.NEUTRAL, 3 )
                    
                    for contains_phrase in self.contains_phrases:
                        # The Y CONTAINS the X. Neither
                        hypothesis = f"The {sm_cont} {contains_phrase} the {sm_obj}."
                        yield ( premise, hypothesis, self.NEUTRAL, 4 )

                        # The Z CONTAINS the X. Neither
                        hypothesis = f"The {med_cont} {contains_phrase} the {sm_obj}."
                        yield ( premise, hypothesis, self.NEUTRAL, 5 )

                        # The Z CONTAINS the Y. Neither
                        hypothesis = f"The {med_cont} {contains_phrase} the {sm_cont}."
                        yield ( premise, hypothesis, self.NEUTRAL, 6 )

    def gen_movement_into_container(self):
        """
        P: The apples are placed in buckets.
        H1: The apples are located inside buckets.
        H2: The apples are located outside buckets.
        H3: The apples are located near buckets.
        """
        for entity, movement_rel, container in product(self.plural_objects, self.place_into_verb, self.containers):
            premise = f"{entity} are {movement_rel} into {container}."
            for in_rel in self.in_rels:
                yield ( premise, f"{entity} are {in_rel} {container}.", self.ENTAILMENT, 0 )
                yield ( premise, f"{container} is {in_rel} {entity}.", self.CONTRADICTION, 1 )
            for outside_rel in self.outside_rels:
                yield ( premise, f"{entity} are {outside_rel} {container}.", self.CONTRADICTION, 2 )

    def gen_filled_with(self):
        """
        P: The basket is filled with apples.
        H1: Apples are on the inside of the basket.
        H2: The basket is on the inside of apples.
        """
        for container, object, in_rel in product(self.containers, self.plural_objects, self.in_rels):
            premise = f"{container} is filled with {object}."
            yield ( premise, f"{object} are physically {in_rel} {container}.", self.ENTAILMENT, 0 )
            yield ( premise, f"{container} is physically {in_rel} {object}.", self.CONTRADICTION, 1 )

    def gen_motion_out_of(self):
        """
        P: John will get out of the car.
        H1: John is on the inside of the car.
        H2: The car is on the inside of John.
        H3: John is on the outside of the car.
        H2: The car is on the outside of John.
        """
        containing_body = {'car', 'elevator', 'building'}
        for agent, containing_body in product(self.agents, containing_body):
            premise = f"{agent} will get out of the {containing_body}."
            for in_rel in self.in_rels:
                yield ( premise, f"{agent.title()} is {in_rel} the {containing_body}.", self.ENTAILMENT, 0 )
                yield ( premise, f"The {containing_body} is {in_rel} {agent}.", self.CONTRADICTION , 1 )
            for out_rel in self.outside_rels:
                yield ( premise, f"{agent.title()} is {out_rel} the {containing_body}.", self.CONTRADICTION, 2 )
                yield ( premise, f"The {containing_body} is {out_rel} {agent}.", self.ENTAILMENT, 3 )

