"""
Indirect Object Identification (IOI) Dataset
============================================
Generates test cases for the IOI task.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class IOITestCase:
    clean_prompt: str
    corrupted_prompt: str
    subject_io: str
    subject_s: str
    correct_token: str
    incorrect_token: str
    case_type: str

    @property
    def is_valid(self):
        return True

def get_ioi_test_cases() -> List[IOITestCase]:
    # ABA and BBA grammatical templates
    templates = [
        ("Then, {S} and {IO} went to the hospital. {S} gave a ring to", "ABA"),
        ("When {IO} and {S} went to the store, {S} gave a drink to", "BBA"),
        ("After {S} and {IO} arrived at the park, {S} passed the ball to", "ABA"),
        ("While {IO} and {S} were at the cafe, {S} handed the menu to", "BBA"),
    ]
    
    # Name pairs (these are single tokens in the GPT-2 vocabulary)
    name_pairs = [
        ("John", "Mary"), ("Alice", "Bob"), ("Michael", "Sarah"), 
        ("David", "Emma"), ("Paul", "Anna"), ("James", "Lisa")
    ]
    
    cases = []
    for template, case_type in templates:
        for s, io in name_pairs:
            cases.append(IOITestCase(
                clean_prompt=template.format(S=s, IO=io),
                corrupted_prompt=template.format(S=io, IO=s), # Swap names for corruption
                subject_io=io,
                subject_s=s,
                correct_token=f" {io}",
                incorrect_token=f" {s}",
                case_type=case_type
            ))
    return cases