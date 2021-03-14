"""
create class GENDER to enumerate genders and dictionaries for gender type conversion
"""
import re

from enum import Enum
from spacy.tokens.token import Token
from typing import Dict


class GENDER(Enum):
    """
    Enumerate possible genders.
    Ignore option resolves to words that should be ignored in particular language
    """
    male = 0
    female = 1
    neutral = 2
    unknown = 3
    ignore = 4

"""
Dictionaries for gender type conversion according to the language
"""
SPACY_GENDER_TYPES = {
    "Masc": GENDER.male,
    "Fem": GENDER.female,
    "Neut": GENDER.neutral,  # seen in Dutch spacy
    "Unk": GENDER.unknown
}

#  Winobias gender type conversion
WB_GENDER_TYPES = {
    "male": GENDER.male,
    "female": GENDER.female,
    "neutral": GENDER.neutral,
}

PYMORPH_GENDER_TYPES = {
    "masc": GENDER.male,
    "femn": GENDER.female,
    "neut": GENDER.neutral,
    None: GENDER.neutral
}


# these two functions are used by spacy_support and german
def get_morphology_dict(token: Token) -> Dict:
    """
    Parse a morphology dictionary from spacy token.
    """
    if not token.morph:
        return {'Gender': 'Unk'}

    morphology = str(token.morph)  # extract a token like this : Gender=Masc|Number=Sing

    #print(morphology)
    #print([prop.split("=") for prop in morphology.split("|")])
    morphology_dict = dict([prop.split("=") for prop in morphology.split("|")])
    #print('dict {}'.format(morphology_dict))
    return morphology_dict


def get_gender_from_token(token: Token):
    """
    Get gender indication from spacy token, if it exists
    """
    # Weird spacy bug? "au" should be male
    if (token.lang_ == "fr") and (token.text == "au") and (token.tag_.startswith("DET")):
        return GENDER.male

    # Italian spacy doesn't seem to split correctly
    if (token.lang_ == "it") and (token.text.startswith("dell'")):
        return GENDER.male

    morph_dict = get_morphology_dict(token)
    if "Gender" not in morph_dict:
        return None

    morph_gender = SPACY_GENDER_TYPES[morph_dict["Gender"]]
    return morph_gender


if __name__ == "__main__":
    pass
