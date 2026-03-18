""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_punctuation = ' ;:,.!?¡¿—…"«»“” '
_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
# Crucial: We add the apostrophe explicitly for Uzbek o' and g'
_uzbek_modifiers = "'" 

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_uzbek_modifiers)

# Special symbol ids
SPACE_ID = symbols.index(" ")
