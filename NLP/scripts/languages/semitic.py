"""
In this code, we aim to identify the gender according to the hebrew and arabic determiners
"""
from spacy.lang.he import Hebrew

# Local imports
from util import GENDER


class HebrewPredictor:
    """
    Hebrew morphology heurstics.
    """
    def __init__(self):
        """
        Init tokenizer for Hebrew.
        """
        self.lang = "he"
        self.cache = {}    # Store calculated professions genders
        self.tokenizer = Hebrew().tokenizer

    def get_gender(self, profession: str, translated_sent=None, entity_index=None, ds_entry=None) -> GENDER:
        """
        Predict gender of an input profession.
        """
        if profession not in self.cache:
            self.cache[profession] = self._get_gender(profession)  # predict gender

        return self.cache[profession]

    def _get_gender(self, profession: str) -> GENDER:
        """
        Predict gender, without using cache.
        Super hacky, based on the last letter.
        """
        if not profession.strip():
            # Empty string is an error in alignment
            return GENDER.unknown

        toks = [w.text for w in self.tokenizer(profession)  # get all profession's tokens which are diffrent of "את"
                if w.text != "את"]

        if any([tok[-1] in ["ת", "ה"] for tok in toks]):  # check female determiners
            return GENDER.female

        return GENDER.male


class ArabicPredictor:
    """
    Arabic morphology heurstics.
    """
    def __init__(self):
        """
        Init tokenizer for Arabic.
        """
        self.lang = "ar"
        self.cache = {}    # Store calculated professions genders
        self.tokenizer = lambda sent: sent.split()  # TODO: Might be better with a dedicated Arabic tokenizer

    def get_gender(self, profession: str, translated_sent=None, entity_index=None, ds_entry=None) -> GENDER:
        """
        Predict gender of an input profession.
        """
        if profession not in self.cache:
            self.cache[profession] = self._get_gender(profession)  # predict gender

        return self.cache[profession]

    def _get_gender(self, profession: str) -> GENDER:
        """
        Predict gender, without using cache.
        Super hacky, based on the last letter being tāʾ marbūṭa.
        """
        if not profession.strip():
            # Empty string is an error in alignment
            return GENDER.unknown

        toks = self.tokenizer(profession)

        if any([("ة" in tok) or ("ﺔ" in tok) for tok in toks]):  # check the female determiners from profession's tokens
            return GENDER.female

        return GENDER.male


if __name__ == "__main__":
    ds = ['female', 1, 'The developer argued with the designer because she did not like the design.', 'developer']
    tr_sent = 'جادل المطور مع المصمم لأنها لم تحب التصميم'
    p = ArabicPredictor()
    pred = p.get_gender('المطور')
    print(pred)
    print("ARABIC TEST DONE")

    ds = ['female', 1, 'The developer argued with the designer because she did not like the design.', 'developer']
    tr_sent = 'היזם התווכח עם המעצב כי היא לא אוהבת את העיצוב'
    p = HebrewPredictor()
    pred_he = p.get_gender('העיצוב')
    print(pred_he)
    print("HEBREW TESTDONE")
