import sys
sys.path.append('../')
sys.path.append('../../src/')
import itertools
from problem import ILPProblem
from language import Language
from logic import Const, FuncSymbol, Predicate, Atom, FuncTerm, Clause, Var


def make_time(y, z, time_func):
    return FuncTerm(time_func, [Const(y), Const(z)])


class BasicCausalProblem(ILPProblem):
    def __init__(self, n=50, noise_rate=0.0):
        self.name = "basic_causal"
        self.pos_examples = []
        self.neg_examples = []
        self.backgrounds = []
        self.init_clauses = []
        p_ = Predicate('.', 1)
        false = Atom(p_, [Const('__F__')])
        true = Atom(p_, [Const('__T__')])
        self.facts = [false, true]
        self.lang = None
        self.noise_rate = noise_rate
        self.n = n
        self.entities = [
            'a', 'b', 'c', 'd', 'e',
            'aa', 'bb', 'cc', 'dd', 'ee',
            'ab', 'bc', 'cd', 'de', 'ea',
        ]
        self.times = [str(x) for x in range(0, 21)]
        self.after_times = [str(x) for x in range(1, 21)]
        pairs = list(itertools.product(self.entities, self.entities))
        self.causes = pairs[:20]

    def get_pos_examples(self):
        effect = self.preds[0]
        time_func = self.funcs[0]
        positives = []
        for x, y in self.causes:
            for z in self.after_times:
                atom = Atom(effect, [Const(x), make_time(y, z, time_func)])
                positives.append(atom)
        self.pos_examples = positives[:self.n]

    def get_neg_examples(self):
        effect = self.preds[0]
        time_func = self.funcs[0]
        positives = set(
            (x, y, z) for x, y in self.causes for z in self.after_times
        )
        candidates = []
        for x, y in itertools.product(self.entities, self.entities):
            for z in self.times:
                if (x, y, z) in positives:
                    continue
                candidates.append((x, y, z))
        for x, y, z in candidates[:self.n]:
            atom = Atom(effect, [Const(x), make_time(y, z, time_func)])
            self.neg_examples.append(atom)

    def get_backgrounds(self):
        cause = self.preds[1]
        after = self.preds[2]
        time_func = self.funcs[0]
        for x, y in self.causes:
            atom = Atom(cause, [Const(x), make_time(y, '0', time_func)])
            self.backgrounds.append(atom)
        for z in self.after_times:
            atom = Atom(after, [Const('0'), Const(z)])
            self.backgrounds.append(atom)

    def get_clauses(self):
        time_func = self.funcs[0]
        time_term = FuncTerm(time_func, [Var('Y'), Var('Z')])
        clause1 = Clause(Atom(self.preds[0], [Var('X'), time_term]), [])
        self.clauses = [clause1]

    def get_templates(self):
        self.templates = [RuleTemplate(body_num=2, const_num=0),
                          RuleTemplate(body_num=0, const_num=0)]

    def get_language(self):
        self.preds = [
            Predicate('effect', 2),
            Predicate('cause', 2),
            Predicate('after', 2),
        ]
        self.funcs = [FuncSymbol('time', 2)]
        self.consts = [Const(x) for x in self.entities + self.times]
        self.lang = Language(
            preds=self.preds,
            funcs=self.funcs,
            consts=self.consts,
            subs_consts=[Const('0')],
        )


if __name__ == '__main__':
    problem = BasicCausalProblem(n=50, noise_rate=0.0)
    problem.compile()
    problem.save_problem()
