from lambeq.backend.grammar import Ty, Cup, Word
from lambeq.text2diagram import CCGType 
from lambeq import BobcatParser
import spacy, lemminflect, os, re

parser_path = '/Users/tls/Desktop/Work/COMP0267/assignment_5/COMP0267_CW/bobcat'
ccg_parser = BobcatParser(model_name_or_path=parser_path, cache_dir=parser_path)
nlp = spacy.load("en_core_web_sm")

def lemmatise_sent(caption):
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

def sent2tree(df, labels=['caption'], parser=ccg_parser):
    for label in labels:
        new_label = label + '_tree'
        tree_arr = parser.sentences2trees(df[label].tolist(), suppress_exceptions=True)

        processed_trees = []
        for i, tree in enumerate(tree_arr):
            try:
                if tree is not None:
                    processed_tree = tree._resolved().collapse_noun_phrases()
                    processed_trees.append(processed_tree)
                else:
                    print(f"Error parsing sentence {i}")
                    processed_trees.append(None)
            except Exception as e:
                print(f"Error processing tree {i}: {e}")
                processed_trees.append(None)
        df[new_label] = processed_trees
        df = df.dropna(subset=[new_label]).reset_index(drop=True)
    return df

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

# def get_trees_mscoco(df, label='caption', parser=ccg_parser):
#     new_label = label + '_tree'
#     tree_arr = parser.sentences2trees(df[label].tolist(), suppress_exceptions=True)
#     processed_trees = []

#     for i, tree in enumerate(tree_arr):
#         try:
#             if tree is not None:
#                 processed_tree = tree._resolved().collapse_noun_phrases()
#                 processed_trees.append(processed_tree)
#             else:
#                 print(f"Error parsing sentence {i}")
#                 processed_trees.append(None)
#         except Exception as e:
#             print(f"Error processing tree {i}: {e}")
#             processed_trees.append(None)

#     df[new_label] = processed_trees
#     df = df.dropna(subset=[new_label]).reset_index(drop=True)
#     return df

# def get_trees_svo(df, parser=ccg_parser):
#     sentence_arr = list(df['corrected_sentence'])
#     trees = parser.sentences2trees(sentence_arr, suppress_exceptions=True)
#     new_trees = []
#     drop_idx = []
#     for i, tree in enumerate(trees):
#         try:
#             if tree is not None:
#                 new_trees.append(tree._resolved().collapse_noun_phrases())
#             else:
#                 drop_idx.append(i)
#         except Exception as e:
#             print(f"Error processing tree {i}: {e}")
#     df = df.drop(df.index[drop_idx]).reset_index(drop=True)
#     df.insert(len(df.columns), 'tree', new_trees)
#     return df

# def get_trees_aro(df, parser=ccg_parser):
#     pos_sent_arr = list(df['true_caption'])
#     neg_sent_arr = list(df['false_caption'])
#     pos_trees = parser.sentences2trees(pos_sent_arr, suppress_exceptions=True)
#     neg_trees = parser.sentences2trees(neg_sent_arr, suppress_exceptions=True)
#     new_pos_trees = []
#     new_neg_trees = []
#     drop_idx = []
#     for i, (pos_tree, neg_tree) in enumerate(zip(pos_trees, neg_trees)):
#         try:
#             if pos_tree is not None and neg_tree is not None:
#                 new_pos_trees.append(pos_tree._resolved().collapse_noun_phrases())
#                 new_neg_trees.append(neg_tree._resolved().collapse_noun_phrases())
#             else:
#                 drop_idx.append(i)
#         except Exception as e:
#             print(f"Error processing tree {i}: {e}")
#     df = df.drop(df.index[drop_idx]).reset_index(drop=True)
#     df.insert(len(df.columns), 'pos_tree', new_pos_trees)
#     df.insert(len(df.columns), 'neg_tree', new_neg_trees)
#     return df

# ???
# def curry_type(ccg_type):
#     if ccg_type.is_atomic:
#         return ccg_type
    
#     res = curry_type(ccg_type.result)
#     arg = curry_type(ccg_type.argument)     
    
#     if res.is_over and ccg_type.is_over:
#         return CCGType(result=CCGType(result=res.result, 
#                                       direction=ccg_type.direction, 
#                                       argument=arg),
#                        direction=res.direction,
#                        argument=res.argument)
    
#     return CCGType(result=res, direction=ccg_type.direction, argument=arg)

# ???
# def curry_tree(node):
#     node.biclosed_type = curry_type(node.biclosed_type)
#     if not node.is_leaf:
#         for child in node.children:
#             curry_tree(child)
#     return node
