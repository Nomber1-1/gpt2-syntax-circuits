"""
Shared Datasets for Subject-Verb Number Agreement
=================================================
This module generates a comprehensive and balanced dataset for testing
Subject-Verb Number agreement, inspired by the methodology of generating
examples across various linguistic feature permutations.
"""
from dataclasses import dataclass
from typing import List
from itertools import product
import random
from tqdm import tqdm

# Inspired by https://github.com/DavidDemitriAfrica/verb-conjugation-mechinterp/blob/main/dataset.py

@dataclass
class SVNTestCase:
    clean_prompt: str
    corrupted_prompt: str
    correct_token: str
    incorrect_token: str
    case_type: str
    subject: str = ""
    subject_number: str = ""

def _generate_balanced_svn_cases(samples_per_perm=5, seed=42) -> List[SVNTestCase]:
    """
    Generates a balanced dataset of SVN test cases based on permutations of linguistic features.
    
    Features:
    - Plurality: Singular vs. Plural subject
    - Negation: Affirmative vs. Negated verb
    - Prefix: With or without a time prefix (e.g., "Yesterday, ")
    - Subject Type: Pronoun vs. Name
    - Tense: Present vs. Past (only SVN-relevant cases)
    - Verb Type: Regular vs. Irregular
    """
    random.seed(seed)
    
    # --- Data for generation ---
    singular_names = [
        "Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", "Hank",
        "Ivy", "Jack", "Kara", "Leo", "Mia", "Nina", "Oscar", "Paul", "Quinn", "Rose", "Sam", "Tina"
    ]
    # Use a smaller, non-overlapping subset for plural names to keep list manageable
    plural_names = [f"{n1} and {n2}" for n1 in singular_names[:5] for n2 in singular_names[5:10]]
    singular_pronouns = ["She", "He"]
    plural_pronouns = ["They"]
    
    past_prefixes = ["Yesterday, ", "Last week, "]
    present_prefixes = ["Today, ", "In the morning, ", "At night, "]
    
    # Note: some of these are technically irregular, but treated as regular stems here.
    reg_present_singular = ["walks", "runs", "talks", "jumps", "sleeps", "writes", "reads", "sings", "dances"]
    reg_present_plural   = ["walk", "run", "talk", "jump", "sleep", "write", "read", "sing", "dance"]
    
    irregular_verbs = {
        "go": {"present_singular": "goes", "present_plural": "go"},
        "eat": {"present_singular": "eats", "present_plural": "eat"},
        "have": {"present_singular": "has", "present_plural": "have"},
        "be": {"present_singular": "is", "present_plural": "are", "past_singular": "was", "past_plural": "were"}
    }
    
    # --- Permutations ---
    conditions = list(product([False, True],    # is_plural
                              [False, True],    # is_negated
                              [False, True],    # has_prefix
                              [False, True],    # is_pronoun
                              ["present", "past"],  # tense
                              [False, True]))   # use_irregular
    
    test_cases = []
    
    for is_plural, is_negated, has_prefix, is_pronoun, tense, use_irregular in tqdm(conditions, desc="Generating SVN cases"):
        
        # --- Filter for SVN-relevant cases ---
        # Present tense is always SVN-relevant.
        # Past tense is only SVN-relevant for the verb 'be'.
        # We also skip negated 'be' as it doesn't fit the "prompt + token" structure well (e.g., "is not").
        if tense == "past" and (not use_irregular or is_negated):
            continue
        
        for _ in range(samples_per_perm):
            # 1. Select subjects (clean and corrupted to swap plurality)
            if is_pronoun:
                clean_subject = random.choice(plural_pronouns if is_plural else singular_pronouns)
                corrupted_subject = random.choice(singular_pronouns if is_plural else plural_pronouns)
            else:
                clean_subject = random.choice(plural_names if is_plural else singular_names)
                corrupted_subject = random.choice(singular_names if is_plural else plural_names)

            # 2. Select prefix
            prefix = ""
            if has_prefix:
                prefix = random.choice(past_prefixes if tense == "past" else present_prefixes)
            
            # 3. Determine prompts and verbs
            clean_prompt, corrupted_prompt = "", ""
            correct_verb, incorrect_verb = "", ""
            
            # Determine auxiliaries for negation (only for present tense)
            aux_singular, aux_plural = "", ""
            if is_negated and tense == "present":
                aux_singular = "does not "
                aux_plural = "do not "
            
            clean_aux = aux_plural if is_plural else aux_singular
            corrupted_aux = aux_singular if is_plural else aux_plural
            
            # --- Verb Logic ---
            if tense == "present":
                verb_key = None
                if use_irregular:
                    if is_negated: # Skip negated 'be'
                        verb_key = random.choice([k for k in irregular_verbs.keys() if k != 'be'])
                    else:
                        verb_key = random.choice(list(irregular_verbs.keys()))
                    forms = irregular_verbs[verb_key]
                    
                    if verb_key == 'be':
                        correct_verb, incorrect_verb = (forms["present_plural"], forms["present_singular"]) if is_plural else (forms["present_singular"], forms["present_plural"])
                        clean_prompt, corrupted_prompt = f'{prefix}{clean_subject} ', f'{prefix}{corrupted_subject} '
                    else:
                        correct_verb = forms["present_plural"] if is_negated or is_plural else forms["present_singular"]
                        incorrect_verb = forms["present_singular"] if is_negated or is_plural else forms["present_plural"]
                        clean_prompt, corrupted_prompt = f'{prefix}{clean_subject} {clean_aux}', f'{prefix}{corrupted_subject} {corrupted_aux}'
                else: # regular
                    verb_idx = random.randrange(len(reg_present_plural))
                    correct_verb = reg_present_plural[verb_idx] if is_negated or is_plural else reg_present_singular[verb_idx]
                    incorrect_verb = reg_present_singular[verb_idx] if is_negated or is_plural else reg_present_plural[verb_idx]
                    clean_prompt, corrupted_prompt = f'{prefix}{clean_subject} {clean_aux}', f'{prefix}{corrupted_subject} {corrupted_aux}'

            elif tense == "past": # Only for non-negated, irregular 'be'
                forms = irregular_verbs["be"]
                correct_verb, incorrect_verb = (forms["past_plural"], forms["past_singular"]) if is_plural else (forms["past_singular"], forms["past_plural"])
                clean_prompt, corrupted_prompt = f'{prefix}{clean_subject} ', f'{prefix}{corrupted_subject} '
            
            # --- Construct Case ---
            case_type = f"{tense}_{'pl' if is_plural else 'sg'}"
            case_type += f"_{'neg' if is_negated else 'aff'}"
            case_type += f"_{'pro' if is_pronoun else 'name'}"
            case_type += f"_{'irr' if use_irregular else 'reg'}"
            case_type += f"_{'pre' if has_prefix else 'nopre'}"

            test_cases.append(SVNTestCase(
                clean_prompt=clean_prompt.strip(),
                corrupted_prompt=corrupted_prompt.strip(),
                correct_token=f" {correct_verb}",
                incorrect_token=f" {incorrect_verb}",
                case_type=case_type,
                subject=clean_subject,
                subject_number='plural' if is_plural else 'singular'
            ))
    
    return test_cases

# Cached global for generated test cases
_cached_svn_test_cases = None

def get_svn_test_cases(limit: int = None) -> List[SVNTestCase]:
    """
    Returns a comprehensive and balanced list of SVN test cases.
    The list is cached after the first generation.
    
    Args:
        limit: If provided, returns only the first `limit` test cases.
    """
    global _cached_svn_test_cases
    if _cached_svn_test_cases is None:
        # samples_per_perm can be adjusted for a larger/smaller test suite
        _cached_svn_test_cases = _generate_balanced_svn_cases(samples_per_perm=5, seed=42)
    
    return _cached_svn_test_cases[:limit] if limit is not None else _cached_svn_test_cases

def get_svn_dataset_prompts() -> List[str]:
    """
    Returns a list of prompts for building activation distributions.
    These are derived from the main test cases to ensure distribution matching.
    """
    test_cases = get_svn_test_cases()
    # Return both clean and corrupted prompts to get a wide distribution
    prompts = [tc.clean_prompt for tc in test_cases] + [tc.corrupted_prompt for tc in test_cases]
    return list(set(prompts)) # Return unique prompts
