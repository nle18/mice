from typing import List, Set

import re

# abstract classes
class Rule:
    def __init__(self) -> None:
        return None

    def is_format(self, span: str) -> bool:
        """Whether the rule applied for this span"""
        raise NotImplementedError

    def generate_surface_forms(self, span: str) -> Set[str]:
        """Generate equivalent spans"""
        raise NotImplementedError

    def span_equal(self, span1: str, span2: str) -> bool:
        """Check if two spans are equivalent"""
        return span2 in self.generate_surface_forms(span1)


# list of rules
class Rule1(Rule):
    """
    This rule considers the following span formats:
    - 'x of y' -> {'x of y', 'y (x)', 'y'}
    - 'y (x)' -> {'x of y', 'y (x)', 'y'}
    Examples:
    - '50 ml of water' -> {'50 ml of water', 'water (50 ml)', 'water'}
    - 'water (50 ml)' -> {'50 ml of water', 'water (50 ml)', 'water'}
    """

    def __init__(self) -> None:
        return None

    def is_format(self, span: str) -> bool:
        # case "50 ml of water"
        if "of" in span and len(span.split(" of ")) == 2:
            return True
        # case  "water (50 ml)"
        elif "(" in span and span[-1] == ")":
            return True
        return False

    def generate_surface_forms(self, span: str) -> Set[str]:
        # first "50 ml of water" -> {'50 ml of water', 'water (50 ml)', 'water'}
        surface_forms = [span]
        if "of" in span and len(span.split(" of ")) == 2:

            quantity, chemical = span.split(" of ")
            surface_forms.append("%s (%s)" % (chemical, quantity))  # form 1
            surface_forms.append(chemical)  # form 2

        # second "water (50 ml)" -> {'50 ml of water', 'water (50 ml)', 'water'}
        elif "(" in span and span[-1] == ")":
            # print(mention)
            # pattern matching
            p = re.compile("(.*) \((.*)\)")
            result = p.search(span)
            try:
                chemical = result.group(1)
                quantity = result.group(2)
                surface_forms.append("%s of %s" % (quantity, chemical))
                surface_forms.append(chemical)
            except:  # wrong matches
                return surface_forms

        return surface_forms


class ChemuRefVerbalizer:
    def __init__(self) -> None:
        self.rules = [Rule1()]

    def filter_span(self, raw_spans: List[str], context: str, anaphor: str) -> Set[str]:
        """Filter out irrelevant spans, based on heuristics and biases"""

        # first filtered out long spans
        filtered_spans = set()
        for span in raw_spans:
            # TODO: Filter out mentions in the title sentence (first sentence)
            if span == context:
                continue
            elif len(span) > len(context) * 3 / 4:
                continue
            elif len(span) > 250:  # range between [200 - 300]
                continue
            elif span in anaphor or anaphor in span:
                continue

            # if spans not in the context, then we apply the rules
            elif span not in context:

                # check if span is_format with rules
                for rule in self.rules:
                    if rule.is_format(span):

                        # generate the list of surface_forms
                        for form in rule.generate_surface_forms(span):

                            # if at least one of the surface forms is in the context, then
                            # add this span to the list
                            if form in context:
                                filtered_spans.add(span)
                                break
            else:
                filtered_spans.add(span)

        return filtered_spans

    def create_span_clusters(self, spans: Set[str], context: str) -> dict:
        """Convert a set of (filtered) spans into clusters of span, based on a set
        of rules
        Example 1:
        raw_spans = {"water (50 ml)", "water", "sodium"}
        outputs = {{"water (50 ml)", "water"}, {"sodium"}}
        Example 2:
        raw_spans = {"water (50 ml)", "water", "sodium", "a solution of water (50 ml) and sodium"}
        outputs = {{"water (50 ml)", "water", "sodium", "a solution of water (50 ml) and sodium"}}

        Algo: nested loops through the filtered spans. If the spans are (1) within another
        or (2) equivalent, then create clusters. We prioritize within another first

        """
        spans = list(spans)
        free_spans = {span: True for span in spans}
        # get the list of "canonical_spans" (unique spans and spans in context)
        canonical_spans = []
        for i in range(len(spans)):
            isUnique = True
            span_i = spans[i]

            for j in range(len(spans)):
                if i != j:
                    span_j = spans[j]
                    if span_i in span_j and span_j in context:
                        isUnique = False
                        break
            if isUnique and span_i in context:
                canonical_spans.append(span_i)

        # sort according to length, to prioritize longer spans
        canonical_spans = sorted(canonical_spans, key=lambda s: len(s), reverse=True)
        # print("spans=", spans)
        # print("canonical_spans=", canonical_spans)
        clusters = dict()
        for canonical_span in canonical_spans:
            cluster = {canonical_span}
            free_spans[canonical_span] = False
            for span in free_spans:
                if free_spans[span]:

                    # prioritize collapsing rules
                    if span in canonical_span:
                        cluster.add(span)

                    # then we do the paraphrase rules
                    else:
                        for rule in self.rules:
                            if span in rule.generate_surface_forms(canonical_span):
                                cluster.add(span)
                    free_spans[span] = False
            clusters[canonical_span] = cluster

        # finally we check for the leftover free spans. This is the case where
        # LM generates something like "water (50 ml)" and not "50 ml of water".
        # In this case, we want to get one of the surface_forms
        for span, is_available in free_spans.items():
            if is_available:
                for rule in self.rules:
                    surface_forms = rule.generate_surface_forms(span)
                    found = False
                    for form in surface_forms:
                        if form in context:
                            clusters[form] = surface_forms
                            found = True
                            break
                    if found:
                        break

        return clusters
