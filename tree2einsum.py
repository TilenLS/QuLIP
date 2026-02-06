from lambeq.text2diagram import CCGType, CCGRule 
from itertools import count

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


def tree2einsum(root_node):
    idx_gen = count(0)
    def get_new_index():
        new_index = next(idx_gen)
        return new_index
    
    root_idx = get_new_index()
    stack = [(root_node, [root_idx])]
    tn = {}

    while stack: 
        node, idx_arr = stack.pop()
        if node.rule == CCGRule.LEXICAL:
            tn[node.text] = (idx_arr, get_type(node.biclosed_type))
        elif node.rule == CCGRule.FORWARD_APPLICATION:
            shared_idx = [get_new_index() for _ in get_type(node.right.biclosed_type)]
            stack.append((node.right, shared_idx))
            stack.append((node.left, idx_arr + shared_idx[::-1]))
        elif node.rule == CCGRule.BACKWARD_APPLICATION:    
            shared_idx = [get_new_index() for _ in get_type(node.left.biclosed_type)]
            stack.append((node.right, shared_idx[::-1] + idx_arr))
            stack.append((node.left, shared_idx))
        elif node.rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
            stack.append(node.right, idx_arr)
        elif node.rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
            stack.append(node.left, idx_arr)

    return tn
