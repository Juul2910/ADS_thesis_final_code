
import re

CONTROLLED_TERMS = {
    "sex": {
        "women's": "female respondent",
        "men's": "male respondent",
        "not specified": "respondent",
        "neutral": "respondent",
    },
    "age_group": {
        "18-29": "young adult respondent",
        "30-39": "adult respondent",
        "40-49": "adult respondent",
        "50-59": "older adult respondent",
        "60-69": "older respondent",
        "70-79": "older respondent",
        "80+": "older respondent",
        "not specified": "respondent",
    },
    "idp_status": {
        "yes": "IDP respondent",
        "no": "non-IDP respondent",
    },
    "invalidity": {
        "person with disability": "respondent with disability",
        "no group": "respondent",
    },
}




def normalize_feedback_entries(text: str) -> str:
    """
    Replaces demographic terms with controlled vocabulary.

    This is intentionally conservative:
    - only replaces known terms
    - does not invent attributes
    - preserves feedback content
    """

    normalized = text

    # Sex
    normalized = re.sub(
        r"\bwomen'?s\b",
        "female respondent",
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = re.sub(
        r"\bmen'?s\b",
        "male respondent",
        normalized,
        flags=re.IGNORECASE,
    )

    # Age
    age_map = {
        "18-29": "young adult respondent",
        "30-39": "adult respondent",
        "40-49": "adult respondent",
        "50-59": "older adult respondent",
        "60-69": "older respondent",
        "70-79": "older respondent",
        "80+": "older respondent",
    }

    for old, new in age_map.items():
        normalized = normalized.replace(old, new)

    # IDP
    normalized = re.sub(
        r"\bIDP\b",
        "IDP respondent",
        normalized,
        flags=re.IGNORECASE,
    )

    # Disability
    normalized = re.sub(
        r"person with disability",
        "respondent with disability",
        normalized,
        flags=re.IGNORECASE,
    )

    return normalized