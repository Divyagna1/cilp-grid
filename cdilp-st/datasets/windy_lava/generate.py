import json
import sys
sys.path.append('../')
sys.path.append('../../src/')
from problem import ILPProblem
from language import Language
from logic import Const, FuncSymbol, Predicate, Atom, Clause, Var


def to_const(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class WindyLavaProblem(ILPProblem):
    def __init__(self, json_path, noise_rate=0.0):
        self.name = "windy_lava"
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
        self.json_path = json_path

    def get_pos_examples(self):
        positives, _ = self._build_examples()
        self.pos_examples = positives

    def get_neg_examples(self):
        _, negatives = self._build_examples()
        self.neg_examples = negatives

    def _build_examples(self):
        positives = []
        negatives = []
        with open(self.json_path) as f:
            data = json.load(f)
        for traj_idx, traj in enumerate(data):
            traj_id = f"tr{traj_idx}"
            terminal_by_t = {}
            wind_by_t = {}
            for tr in traj.get('transitions', []):
                t = tr['step_index']
                terminal_by_t[t] = tr.get('terminated', False)
                wind_by_t[t] = to_const(tr['observation']['wind'])
            max_t = max(terminal_by_t.keys()) if terminal_by_t else -1
            for t in range(max_t + 1):
                t2 = t + 2
                if t2 not in terminal_by_t:
                    continue
                wind = wind_by_t.get(t)
                if wind is None:
                    continue
                atom = Atom(self.preds[0], [
                    Const(traj_id),
                    Const(str(t)),
                    Const(str(wind)),
                    Const(str(t2)),
                    Const('2'),
                ])
                if terminal_by_t[t2]:
                    positives.append(atom)
                else:
                    negatives.append(atom)
        return positives, negatives

    def get_backgrounds(self):
        windat = self.preds[1]
        agentat = self.preds[2]
        lavadistance = self.preds[3]
        terminal = self.preds[4]
        stepadd = self.preds[5]
        with open(self.json_path) as f:
            data = json.load(f)
        max_t = -1
        for traj_idx, traj in enumerate(data):
            traj_id = f"tr{traj_idx}"
            for tr in traj.get('transitions', []):
                t = tr['step_index']
                max_t = max(max_t, t)
                obs = tr['observation']
                agent = obs['agent']
                wind = obs['wind']
                dist = tr['info']['distance']
                self.backgrounds.append(
                    Atom(windat, [Const(traj_id), Const(str(t)), Const(to_const(wind))])
                )
                self.backgrounds.append(
                    Atom(agentat, [
                        Const(traj_id), Const(str(t)), Const(to_const(agent[0])), Const(to_const(agent[1]))
                    ])
                )
                self.backgrounds.append(
                    Atom(lavadistance, [Const(traj_id), Const(str(t)), Const(to_const(dist))])
                )
                if tr.get('terminated', False):
                    self.backgrounds.append(
                        Atom(terminal, [Const(traj_id), Const(str(t))])
                    )
        if max_t >= 0:
            for t in range(max_t + 1):
                t2 = t + 2
                self.backgrounds.append(
                    Atom(stepadd, [Const(str(t)), Const('2'), Const(str(t2))])
                )

    def get_clauses(self):
        clause1 = Clause(
            Atom(self.preds[0], [Var('X'), Var('T'), Var('W'), Var('Y'), Var('Z')]),
            []
        )
        self.clauses = [clause1]

    def get_language(self):
        self.preds = [
            Predicate('diesintwo', 5),
            Predicate('windat', 3),
            Predicate('agentat', 4),
            Predicate('lavadistance', 3),
            Predicate('terminal', 2),
            Predicate('stepadd', 3),
        ]
        consts = set()
        with open(self.json_path) as f:
            data = json.load(f)
        max_t = -1
        for traj_idx, traj in enumerate(data):
            traj_id = f"tr{traj_idx}"
            consts.add(traj_id)
            for tr in traj.get('transitions', []):
                t = tr['step_index']
                max_t = max(max_t, t)
                consts.add(str(t))
                obs = tr['observation']
                agent = obs['agent']
                consts.add(to_const(agent[0]))
                consts.add(to_const(agent[1]))
                consts.add(to_const(obs['wind']))
                consts.add(to_const(tr['info']['distance']))
        if max_t >= 0:
            for t in range(max_t + 3):
                consts.add(str(t))
        consts.add('2')
        consts.add('0')
        all_consts = [Const(c) for c in sorted(consts)]
        self.lang = Language(
            preds=self.preds,
            funcs=[FuncSymbol('f', 0)],
            consts=all_consts,
            subs_consts=[Const('2')],
        )


if __name__ == '__main__':
    json_path = '../../../causalrl-main/windy_lava_trajectories.json'
    problem = WindyLavaProblem(json_path=json_path, noise_rate=0.0)
    problem.compile()
    problem.save_problem()
