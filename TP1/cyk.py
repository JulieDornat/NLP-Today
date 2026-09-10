
class Node():
    def __init__(self, type, left, right = None, word=None):
        self.type = type
        self.left = left
        self.right = right
        self.word = word

    def _recursive_repr(self, tab=0):
        s =  f"{'----' * tab}{self.type}"
        if self.word is not None:
            s += f" - {self.word}"
        s+= "\n"
        if self.left is not None:
            s+=  f"{self.left._recursive_repr(tab+1)}"
        if self.right is not None:
            s+=  f"{self.right._recursive_repr(tab+1)}"
        return s

    def __str__(self):
        return self._recursive_repr() + "\n"

    def __repr__(self,):
        return self._recursive_repr() + "\n"

class CYK():
    
    def __init__(self, grammar):
        """ A simple implementation of CYK algorithm
            Parameter:
                grammar : list
                    a list of production rule in chumsky normal form
                    each rule is defined by a tuple (P, (L,R)) with P
                    a non-terminal and L and R respectivelly the right 
                    and left of the production.

            Usage example: 
                sentence = ["The", "cat", "sat",  "on", "the", "couch"]
                pos =      [["DET"], ["NOUN"], ["VERB"], ["PREP"], ["DET"], ["NOUN"]]

                # defining the grammar rules
                G = [
                    # S is axiom
                    ("S", ("NP", "VP")), 
                    
                        # non terminal rules
                    ("NP", ("DET", "NOUN")),
                    ("PP", ("PREP", "NP")),
                    ("VP", ("VERB", "PP")),
                    
                    # terminal (if using pos directly)
                    ("DET", ("DET",)),
                    ("VERB", ("VERB",)),
                    ("NOUN", ("NOUN",)),
                    ("PREP", ("PREP",))
                ]
                cyk = CYK(G)
                cyk(pos, sentence) 

                ## SHOULD RETURN
                [S
                ----NP
                --------DET - The
                --------NOUN - cat
                ----VP
                --------VERB - sat
                --------PP
                ------------PREP - on
                ------------NP
                ----------------DET - the
                ----------------NOUN - couch
                ]
        """
        self.grammar = grammar

    
    def __call__(self, input_list, str):
        """ Apply the CYK algorithm on an input.

            Parameter: 
                input_list: the list of terminal (in our case a list)

            Return:
                List of tree that are recognized by the language
        """
        size = len(input_list)
        cyk_tab = [ [[] for _ in range(i+1)] for i in range(len(input_list)) ][::-1]
        
        # process terminals
        for i, tags_list in enumerate(input_list):
            for tag in tags_list:
                # for all terminal rules
                for production, rule in self.grammar: 
                    if(len(rule) == 1) and (tag in rule):
                        cyk_tab[0][i].append(Node(production, None, word=str[i]))
        for cyk_depth in range(2, size+1):
            left_levels = list(range(1, cyk_depth))
            right_levels = list(range(1, cyk_depth)[::-1])
            for start in range(0, size - cyk_depth + 1):
                for left_level, right_level in zip(left_levels, right_levels):
                    left_start = start
                    right_start = start + left_level
                    Al, Bl = cyk_tab[left_level-1][left_start], cyk_tab[right_level-1][right_start]
                    for p, j in self.grammar:
                        if(len(j) == 2):
                            l, r = j
                            A = [a for a in Al if(a.type == l)]
                            B = [b for b in Bl if(b.type == r)]
                            cyk_tab[cyk_depth-1][start] += [Node(p, a, b ) for a in A  for b in B]

        return [i for i in cyk_tab[-1][0]]
