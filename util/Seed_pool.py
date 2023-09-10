import ast
import heapq
import os

import astunparse
import numpy as np

from util.instrumentor import (
    DepthFinder,
    MultiKeywordFinder,
    SnippetInfill,
    SnippetInfillArbitratyAPI,
    UniqueFinder,
)
from util.util import load_api_symbols


class GA(object):
    def __init__(
        self,
        initial_seeds,
        num_selection,
        num_generated,
        folder,
        api_call,
        mask_identifier,
        library,
        relaxargmut,
        seed_selection_algo="fitness",
        mutator_selection_algo="heuristic",
        use_single_mutator=False,
        replace_type=None,
        seed_pool_size=30,
        mutator_set="all",
    ):
        """
        args:
            seed_selection_algo:
                fitness: weighted-features
                random: uniform random

            mutator_selection_algo:
                heuristic: heuristic functions
                random: uniform random
                epsgreedy: Epsilon Greedy (eps=0.1)
                ucb: Upper Confidence Bound
        """
        self.num_generated = num_generated
        self.num_selection = num_selection
        self.api_call = api_call
        self.mask_identifier = mask_identifier
        self.folder = folder
        self.library = library
        self.relaxargmut = relaxargmut
        self.seed_selection_algo = seed_selection_algo
        self.mutator_selection_algo = mutator_selection_algo
        self.use_single_mutator = use_single_mutator
        self.replace_type = replace_type
        self.seed_pool_size = seed_pool_size
        self.mutator_set = mutator_set

        self.api_call_list, self.full_api_list = load_api_symbols(library)
        self.recursive_infill = SnippetInfillArbitratyAPI(
            mask_identifier, self.api_call_list, self.full_api_list, self.library
        )
        self.num_valid = 0
        self.library = library

        self._init_seed(initial_seeds)
        self._init_mutator()

    def _init_seed(self, initial_seeds):
        self.seeds = []
        self.info_code = {}

        for idx, seed in enumerate(initial_seeds):
            heapq.heappush(self.seeds, (-self.num_generated / 2, seed))
            self.info_code[seed] = {
                "mutation_layer": 0,
                "used_as_seed": 0,
                "parent": None,
                "filename": "{}_{}{}.py".format(self.api_call, "seed", idx + 1),
            }

    def _init_multi_arm(self):
        self.replace_type_p = {}
        for t in self.replace_type:
            # Initialize with 1 to avoid starving
            # Change initial state to 0.5 following the Thompson Sampling algorithm
            self.replace_type_p[t] = [1, 2]  # success / total
        self.epsilon = 0.1

    def _init_mutator(self):
        if self.use_single_mutator:
            self.replace_type = [self.replace_type]
        else:
            if self.mutator_set == "noprefix":
                self.replace_type = [
                    "argument",
                    "keyword",
                    "method",
                    "suffix",
                    "suffix-argument",
                ]
            elif self.mutator_set == "nosuffix":
                self.replace_type = [
                    "argument",
                    "keyword",
                    "method",
                    "prefix",
                    "prefix-argument",
                ]
            elif self.mutator_set == "noargument":
                self.replace_type = [
                    "method",
                    "prefix",
                    "prefix-argument",
                    "suffix",
                    "suffix-argument",
                ]
            elif self.mutator_set == "nomethod":
                self.replace_type = [
                    "argument",
                    "keyword",
                    "prefix",
                    "prefix-argument",
                    "suffix",
                    "suffix-argument",
                ]
            elif self.mutator_set == "all":
                self.replace_type = [
                    "argument",
                    "keyword",
                    "method",
                    "prefix",
                    "prefix-argument",
                    "suffix",
                    "suffix-argument",
                ]
            else:
                print("Replace_type {self.replace_type} not supported.")
                exit(-1)

        if self.mutator_selection_algo == "heuristic":
            self.replace_type_p = {
                "argument": self.num_generated * 3,
                "keyword": self.num_generated * 3,
                "method": self.num_generated * 3,
                "prefix": self.num_generated * 3,
                "prefix-argument": self.num_generated * 3,
                "suffix": self.num_generated * 3,
                "suffix-argument": self.num_generated * 3,
            }
        elif self.mutator_selection_algo in ["epsgreedy", "ucb", "ts"]:
            # Multi-Arm Bandit strategies
            self._init_multi_arm()

    def _add_new_seed(self, seed, code, replace_type, rd, filename):
        if code not in self.info_code:
            self.num_valid += 1
            self.info_code[code] = {
                "mutation_layer": self.info_code[seed]["mutation_layer"] + 1,
                "used_as_seed": 0,
                "parent": seed,
                "replace_type": replace_type,
                "round": rd,
                "filename": filename,
            }
            heapq.heappush(self.seeds, (-self.num_generated / 2, code))

    def _update_seed(self, code, value):
        if value == 0:
            value = -1
        # negative for max heap
        heapq.heappush(self.seeds, (-value, code))
        self.info_code[code]["used_as_seed"] += self.num_generated

    def _compute_score(self, code):
        raise NotImplementedError

    def _select_mutator(self):
        if self.mutator_selection_algo == "heuristic":
            replace_type = np.random.choice(
                self.replace_type,
                1,
                p=[
                    self.replace_type_p[x] / sum(list(self.replace_type_p.values()))
                    for x in self.replace_type
                ],
            )[0]
            return replace_type
        elif self.mutator_selection_algo == "random":
            replace_type = np.random.choice(self.replace_type)
            return replace_type
        elif self.mutator_selection_algo == "epsgreedy":
            expl = np.random.uniform(0.0, 1.0)
            if expl > self.epsilon:  # exploit
                max_value = max(
                    [x[0] / x[1] for v, x in self.replace_type_p.items() if x[1] != 0],
                    default=0,
                )
                if max_value == 0:
                    replace_type = np.random.choice(self.replace_type)
                else:
                    replace_type = [
                        k
                        for k, v in self.replace_type_p.items()
                        if v[1] != 0 and v[0] / v[1] == max_value
                    ][0]
            else:  # explore
                replace_type = np.random.choice(self.replace_type)
            return replace_type
        elif self.mutator_selection_algo == "ucb":
            total_num = sum([x[1] for x in self.replace_type_p.values()])
            log_t_2 = 2.0 * np.log(total_num)
            # UCB1 score: mu(a) + sqrt(2 * log(t) / n_t(a))
            type_scores = [
                (v, x[0] / x[1] + np.sqrt(log_t_2 / x[1]))
                for v, x in self.replace_type_p.items()
            ]
            types, scores = list(zip(*type_scores))
            max_index = np.argmax(scores)
            max_score = scores[max_index]
            max_types = [t for t, score in type_scores if score >= max_score]
            return np.random.choice(max_types)
        elif self.mutator_selection_algo == "ts":
            scores = []
            for a in self.replace_type:
                alpha, n_t = self.replace_type_p[a]
                beta = n_t - alpha
                score_a = np.random.beta(alpha, beta)
                scores.append(score_a)
            max_index = np.argmax(scores)
            return self.replace_type[max_index]

    def _select_seed(self):
        code = heapq.heappop(self.seeds)[-1]
        return code

    def selection(self):
        selections = []
        while self.seeds and len(selections) != self.num_selection:
            code = self._select_seed()
            replace_type = self._select_mutator()

            infill_code = None
            if (
                (replace_type == "argument" and self.relaxargmut)
                or replace_type == "keyword"
                or replace_type == "method"
            ):
                try:
                    o_ast = ast.parse(code)
                    original_code = astunparse.unparse(o_ast).strip()
                    o_ast = ast.parse(original_code)  # reparse
                except Exception as e:  # if the snippet is not valid python syntax
                    print("Error parsing snippet")
                else:
                    (
                        num_replaced,
                        infill_code,
                        original_code,
                    ) = self.recursive_infill.add_infill(
                        o_ast,
                        add_keywords=(replace_type == "keyword"),
                        replace_method=(replace_type == "method"),
                    )
                    if num_replaced == 0:
                        infill = SnippetInfill(
                            mask_identifier=self.mask_identifier,
                            api_call=self.api_call.split(".")[-1],
                            prefix=".".join(self.api_call.split(".")[1:-1]),
                            library=self.library,
                            replace_type="argument",
                        )
                        num_replaced, infill_code, _ = infill.add_infill(code)
                    assert num_replaced >= 1

            else:
                infill = SnippetInfill(
                    mask_identifier=self.mask_identifier,
                    api_call=self.api_call.split(".")[-1],
                    prefix=".".join(self.api_call.split(".")[1:-1]),
                    library=self.library,
                    replace_type=replace_type,
                )
                num_replaced, infill_code, _ = infill.add_infill(code)
                assert num_replaced >= 1
            if infill_code is not None:
                selections.append((code, infill_code, replace_type))
        return selections

    def _update_mutator(self, generations, replace_type):
        if self.mutator_selection_algo == "heuristic":
            # update the global counter
            # roughly that score increases when at least 1/4 of generation is valid and unique
            self.replace_type_p[replace_type] += len(generations) - 1 / 3 * (
                self.num_generated - len(generations)
            )
            self.replace_type_p[replace_type] = max(
                1, self.replace_type_p[replace_type]
            )
        elif self.mutator_selection_algo in ["epsgreedy", "ucb", "ts"]:
            # update the global counter
            self.replace_type_p[replace_type][0] += len(generations)
            self.replace_type_p[replace_type][1] += self.num_generated
        elif self.mutator_selection_algo == "random":
            pass
        else:
            raise NotImplementedError(
                "Operator selction algorithm {} not supported".format(
                    self.mutator_selection_algo
                )
            )

    def update(self, seed, generations, replace_type, rd, filenames):
        self._update_seed(seed, len(generations))
        self._update_mutator(generations, replace_type)
        for g, filename in zip(generations, filenames):
            self._add_new_seed(seed, g, replace_type, rd, filename)

    def get_highest_order_output(self):
        highest_order = max([v["mutation_layer"] for n, v in self.info_code.items()])
        for n, v in self.info_code.items():
            if v["mutation_layer"] == highest_order:
                max_depth = DepthFinder(self.library).max_depth(n)
                _, _, repeats = UniqueFinder(self.library).count(n)
                keywords = MultiKeywordFinder(self.library).count(n)
                print(max_depth, repeats, keywords)
                return n, highest_order

    def get_p(self):
        if self.mutator_selection_algo == "heutistic":
            return [
                self.replace_type_p[x] / sum(list(self.replace_type_p.values()))
                for x in self.replace_type
            ]
        elif self.mutator_selection_algo == "random":
            return [1.0 / len(self.replace_type)] * len(self.replace_type)
        else:
            return self.replace_type_p


class GA_Random(GA):
    def _init_seed(self, initial_seeds):
        self.seeds = []
        self.info_code = {}

        for idx, seed in enumerate(initial_seeds):
            self.seeds.append(seed)
            self.info_code[seed] = {
                "mutation_layer": 0,
                "used_as_seed": 0,
                "parent": None,
                "filename": "{}_{}{}.py".format(self.api_call, "seed", idx + 1),
            }

    def _update_seed(self, code, value):
        self.info_code[code]["used_as_seed"] += self.num_generated
        self.seeds.append(code)

    def _select_seed(self):
        code = np.random.choice(self.seeds)
        self.seeds.remove(code)
        return code

    def _add_new_seed(self, seed, code, replace_type, rd, filename):
        if code not in self.info_code:
            self.num_valid += 1
            self.info_code[code] = {
                "mutation_layer": self.info_code[seed]["mutation_layer"] + 1,
                "used_as_seed": 0,
                "parent": seed,
                "replace_type": replace_type,
                "round": rd,
                "filename": filename,
            }
            self.seeds.append(code)


class GA_Coverage(GA):
    def _init_seed(self, initial_seeds):
        self.seeds = []
        self.info_code = {}

        for idx, seed in enumerate(initial_seeds):
            self.seeds.append(seed)
            self.info_code[seed] = {
                "mutation_layer": 0,
                "used_as_seed": 0,
                "parent": None,
                "filename": "{}_{}{}.py".format(self.api_call, "seed", idx + 1),
            }

    def _update_seed(self, code, value):
        self.info_code[code]["used_as_seed"] += self.num_generated
        self.seeds.append(code)

    def _select_seed(self):
        code = np.random.choice(self.seeds)
        self.seeds.remove(code)
        return code

    def _add_new_seed(self, seed, code, replace_type, rd, filename):
        if code not in self.info_code:
            self.num_valid += 1
            self.info_code[code] = {
                "mutation_layer": self.info_code[seed]["mutation_layer"] + 1,
                "used_as_seed": 0,
                "parent": seed,
                "replace_type": replace_type,
                "round": rd,
                "filename": filename,
            }
            self.seeds.append(code)

    def update(self, seed, generations, replace_type, rd, filenames, add_flags):
        self._update_seed(seed, len(generations))
        self._update_mutator(generations, replace_type)
        for g, filename, add_flag in zip(generations, filenames, add_flags):
            if add_flag:
                self._add_new_seed(seed, g, replace_type, rd, filename)


class GAR(GA):
    def _add_new_seed(self, seed, code, replace_type, rd, filename):
        if code not in self.info_code:
            self.num_valid += 1
            self.info_code[code] = {
                "mutation_layer": self.info_code[seed]["mutation_layer"] + 1,
                "used_as_seed": 0,
                "parent": seed,
                "replace_type": replace_type,
                "round": rd,
                "filename": filename,
            }
            unique_calls, _, _ = UniqueFinder(self.library).count(code)
            heapq.heappush(self.seeds, (-self.num_generated / 2 - unique_calls, code))

    def _update_seed(self, code, value):
        if value == 0:
            value = -1
        unique_calls, _, _ = UniqueFinder(self.library).count(code)
        # negative for max heap
        heapq.heappush(self.seeds, (-value - unique_calls, code))
        self.info_code[code]["used_as_seed"] += self.num_generated


class GAR_depth(GA):
    def _init_seed(self, initial_seeds):
        self.seeds = []

        self.info_code = {}

        for idx, seed in enumerate(initial_seeds):
            self.info_code[seed] = {
                "mutation_layer": 0,
                "used_as_seed": 0,
                "parent": None,
                "filename": "{}_{}{}.py".format(self.api_call, "seed", idx + 1),
            }
            heapq.heappush(
                self.seeds,
                (self._compute_fitness_score(seed), -len(seed.splitlines()), seed),
            )
        if self.seed_pool_size > 0:
            while len(self.seeds) > self.seed_pool_size:
                heapq.heappop(self.seeds)

    def _compute_fitness_score(self, code):
        if self.seed_selection_algo == "fitness":
            max_depth = DepthFinder(self.library).max_depth(code)
            unique_calls, ex_repeats, repeats = UniqueFinder(self.library).count(code)
            return unique_calls + max_depth - ex_repeats

        elif self.seed_selection_algo == "fitnessue":
            unique_calls, ex_repeats, repeats = UniqueFinder(self.library).count(code)
            return unique_calls - ex_repeats

        elif self.seed_selection_algo == "fitnessud":
            max_depth = DepthFinder(self.library).max_depth(code)
            unique_calls, ex_repeats, repeats = UniqueFinder(self.library).count(code)
            return unique_calls + max_depth

        elif self.seed_selection_algo == "fitnessde":
            max_depth = DepthFinder(self.library).max_depth(code)
            unique_calls, ex_repeats, repeats = UniqueFinder(self.library).count(code)
            return max_depth - ex_repeats

    def _add_new_seed(self, seed, code, replace_type, rd, filename):
        if code not in self.info_code:
            self.num_valid += 1
            self.info_code[code] = {
                "mutation_layer": self.info_code[seed]["mutation_layer"] + 1,
                "used_as_seed": 0,
                "parent": seed,
                "replace_type": replace_type,
                "round": rd,
                "filename": filename,
            }
            heapq.heappush(
                self.seeds,
                (self._compute_fitness_score(code), -len(code.splitlines()), code),
            )
            if self.seed_pool_size > 0:
                while len(self.seeds) > self.seed_pool_size:
                    heapq.heappop(self.seeds)

    def _update_seed(self, code, value):
        self.info_code[code]["used_as_seed"] += self.num_generated

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _select_seed(self):
        _seed_scores = [rec[0] for rec in self.seeds]
        codes = [rec[-1] for rec in self.seeds]
        probs = self._softmax(_seed_scores)
        code = np.random.choice(codes, p=probs)
        return code
