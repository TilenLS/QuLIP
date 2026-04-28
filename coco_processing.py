import pandas as pd
import spacy, re, lemminflect
from tqdm import tqdm
from collections import Counter
from itertools import count 
from lambeq.text2diagram import CCGRule, CCGType

nlp = spacy.load("en_core_web_sm")

def stream2df(data_subset):
    data_list = []
    
    for entry in data_subset:
        data_list.append({
            'cocoid': entry['cocoid'],
            'filename': entry['filename'],
            'caption': entry['caption']
        })
    
    df = pd.DataFrame(data_list)
    return df

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

def get_trees_mscoco(df, label='caption', parser=None):
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

def tree2einsum(root_node, simplify=True):
    if simplify:
        root_node = root_node._resolved().collapse_noun_phrases()
    idx_gen = count(0)
    def get_new_index():
        new_index = next(idx_gen)
        return new_index
    
    root_idx = get_new_index()
    stack = [(root_node, [root_idx])]
    tn = []

    while stack: 
        node, idx_arr = stack.pop()
        if node.rule == CCGRule.LEXICAL:
            # tn[node.text] = (idx_arr, get_type(node.biclosed_type))
            tn.append((node.text, idx_arr, get_type(node.biclosed_type)))
        elif node.rule == CCGRule.FORWARD_APPLICATION:
            if node.left.biclosed_type.to_string() in ['(NP/N)', '(n/n)']:
                stack.append((node.right, idx_arr))
                continue
            shared_idx = [get_new_index() for _ in get_type(node.right.biclosed_type)]
            stack.append((node.right, shared_idx))
            stack.append((node.left, idx_arr + shared_idx[::-1]))
        elif node.rule == CCGRule.BACKWARD_APPLICATION:    
            shared_idx = [get_new_index() for _ in get_type(node.left.biclosed_type)]
            stack.append((node.right, shared_idx[::-1] + idx_arr))
            stack.append((node.left, shared_idx))
        elif node.rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
            stack.append((node.right, idx_arr))
        elif node.rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
            stack.append((node.left, idx_arr))
        elif node.rule == CCGRule.UNARY:
            child_types = get_type(node.children[0].biclosed_type)
            if len(child_types) == len(idx_arr):
                stack.append((node.children[0], idx_arr))
            else:
                extra_idx = [get_new_index() for _ in range(len(child_types) - len(idx_arr))]
                stack.append((node.children[0], idx_arr + extra_idx))
        elif node.rule in (CCGRule.FORWARD_COMPOSITION, 
                           CCGRule.BACKWARD_COMPOSITION,
                           CCGRule.FORWARD_CROSSED_COMPOSITION, 
                           CCGRule.BACKWARD_CROSSED_COMPOSITION):
            N_L = len(get_type(node.left.biclosed_type))
            N_R = len(get_type(node.right.biclosed_type))
            N_res = len(get_type(node.biclosed_type))
            N_Y = (N_L + N_R - N_res) // 2
            shared_idx = [get_new_index() for _ in range(N_Y)]

            if node.rule == CCGRule.FORWARD_COMPOSITION:
                N_X = N_L - N_Y
                idx_X = idx_arr[:N_X]
                idx_Zrev = idx_arr[N_X:]
                stack.append((node.right, shared_idx + idx_Zrev))
                stack.append((node.left, idx_X + shared_idx[::-1]))

            elif node.rule == CCGRule.BACKWARD_COMPOSITION: 
                N_Z = N_L - N_Y
                idx_Zrev = idx_arr[:N_Z]
                idx_X = idx_arr[N_Z:]
                stack.append((node.right, shared_idx[::-1] + idx_X))
                stack.append((node.left, idx_Zrev + shared_idx))

            elif node.rule == CCGRule.FORWARD_CROSSED_COMPOSITION: 
                N_Z = N_R - N_Y
                idx_Zrev = idx_arr[:N_Z]
                idx_X = idx_arr[N_Z:]
                stack.append((node.right, idx_Zrev + shared_idx))
                stack.append((node.left, idx_X + shared_idx[::-1]))

            elif node.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION: 
                N_X = N_R - N_Y
                idx_X = idx_arr[:N_X]
                idx_Zrev = idx_arr[N_X:]
                stack.append((node.right, shared_idx[::-1] + idx_X))
                stack.append((node.left, shared_idx + idx_Zrev))

    return tn

def unify_output(tn):
    word_arr, idx_arr, type_arr = zip(*tn)

    # Get output indices of tensor network
    flat_idx_arr = sum(idx_arr, [])
    count_dict = Counter(flat_idx_arr)
    output_idx = [key for key, val in count_dict.items() if val == 1]

    # Find indices of word containing output indices (root word)
    for i, word_idx in enumerate(idx_arr):
        if set(output_idx).issubset(word_idx):
            break 
    
    # Compute location of output indices inside root word indices
    start_idx = word_idx.index(output_idx[0])
    end_idx = start_idx + len(output_idx) 
    
    # Replace root word output indices and type with single unique output index and type
    idx_arr[i][start_idx:end_idx] = [0]
    type_arr[i][start_idx:end_idx] = ['out']
    return list(zip(word_arr, idx_arr, type_arr))

def tree2tn(df, labels=['caption'], simplify=True):
    einsum_arr = []
    if len(labels) == 1:
        label = labels[0]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            einsum = unify_output(tree2einsum(row[label], simplify=simplify))
            einsum_arr.append(einsum)
    else:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            einsum_batch = [unify_output(tree2einsum(row[label], simplify=simplify)) for label in labels]
            einsum_arr.append(einsum_batch)
    return einsum_arr