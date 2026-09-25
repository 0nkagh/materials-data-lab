"""Steel classification rules for Materials Data Lab."""

def classify_steel(steel_type: str) -> str:
    """
    Classify a steel type string into a standard category based on SAE numbering.
    Returns 'UNKNOWN_CLASS' if it doesn't match standard patterns.
    """
    if not isinstance(steel_type, str):
        return "UNKNOWN_CLASS"
        
    s = steel_type.strip()
    if not s or s == "?" or s.lower() == "nan":
        return "UNKNOWN_CLASS"
        
    # Check for Nitriding Steel explicitly
    if "Nitriding Steel" in s:
        return "nitriding"
        
    # Extract the potential AISI/SAE number. It usually looks like 10xx, 41xx etc.
    # We look for "10", "13", "40", "41", "43", "51", "52", "61", "86", "87", "92" 
    # either standalone or right after AISI-SAE or %C logic.
    
    # We will search for a 4-digit or 5-digit number that represents the grade.
    import re
    # Match any 4 or 5 digit number that could be a grade (e.g., 1020, 10100), including optional 'E' prefix
    match = re.search(r'\bE?(\d{4,5})\b', s)
    if match:
        grade = match.group(1)
    else:
        # Some are written like "0,98%C - plain carbon steel" which is plain carbon
        if "plain carbon" in s.lower():
            return "plain_carbon"
        grade = ""
        
    if grade:
        prefix2 = grade[:2]
        if prefix2 == "10":
            return "plain_carbon"
        elif prefix2 == "13":
            return "Mn_steel"
        elif prefix2 == "40":
            return "Mo_steel"
        elif prefix2 == "41":
            return "Cr_Mo"
        elif prefix2 in ("43", "86", "87"):
            return "Ni_Cr_Mo"
        elif prefix2 in ("51", "52"):
            return "Cr_steel"
        elif prefix2 == "61":
            return "Cr_V"
        elif prefix2 == "92":
            return "Si_steel"

    # Some might just be named plain carbon without a 4-digit number
    if "plain carbon" in s.lower():
        return "plain_carbon"

    return "UNKNOWN_CLASS"
