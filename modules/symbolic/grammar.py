from lambeq.text2diagram import CCGType 
import re, spacy

def get_type(ccgtype):
    res_arr = []
    type_arr = [ccgtype]
    while type_arr: 
        cur_type = type_arr.pop()
        if cur_type.is_over:
            type_arr.append(cur_type.result)
            if cur_type.argument.is_complex: 
                old_arg = cur_type.argument
                new_arg = CCGType(result=old_arg.argument, direction=old_arg.direction, argument=old_arg.result)
                type_arr.append(new_arg)
            else:
                type_arr.append(cur_type.argument)
        elif cur_type.is_under: 
            if cur_type.argument.is_complex:
                old_arg = cur_type.argument
                new_arg = CCGType(result=old_arg.argument, direction=old_arg.direction, argument=old_arg.result)
                type_arr.append(new_arg)
            else:
                type_arr.append(cur_type.argument)
            type_arr.append(cur_type.result)
        else: 
            res_arr.append(cur_type.name)
    return res_arr[::-1]

def strip_sent(caption):
    caption = re.sub(r'[^\w\s]', '', caption)
    return " ".join(caption.split()).lower()

def lemmatise_sent(caption, nlp=spacy.load("en_core_web_sm")):
    caption = re.sub(r'[^\w\s]', '', caption)
    caption = " ".join(caption.split())
    doc = nlp(caption)

    has_formal_verb = any(t.tag_ in ("VBZ", "VBP", "VBD") for t in doc)
    if has_formal_verb:
        return caption.strip().capitalize() + ("." if not caption.endswith(".") else "")
    
    new_tokens = []
    replace = False
    for token in doc:
        is_aux = any(child.dep_ == "aux" for child in token.children)
        if token.tag_ == "VBG" and not is_aux and not replace:
            # 1. Find the subject to determine singular vs plural
            is_plural = False
            for child in token.head.children:
                if child.dep_ in ("nsubj", "nsubjpass") and child.morph.get("Number") == ["Plur"]:
                    is_plural = True
                    break
            
            # 2. Inflect based on number (VBZ for singular, VBP for plural)
            target_tag = "VBP" if is_plural else "VBZ"
            finite_verb = token._.inflect(target_tag)
            new_tokens.append(finite_verb if finite_verb else token.text)
            replace = True
        else:
            new_tokens.append(token.text)

    sentence = " ".join(new_tokens).strip()
    return sentence.capitalize()

def lemmatise_df(df, mode="replace"):
    if mode == "replace":
        df['caption'] = df['caption'].apply(lemmatise_sent)
    elif mode == "augment":
        df['lemmatised'] = df['caption'].apply(lemmatise_sent)
    else:
        raise ValueError("Mode must be 'replace' or 'augment'")
    return df