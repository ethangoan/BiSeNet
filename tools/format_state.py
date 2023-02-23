#!/usr/bin/env python3


def format_state_params(state, model_type):
    print(model_type)
    if model_type == 'bisenetv2':
        updated_state = update_state_bisenetv2(state)
    elif model_type == 'bisenetv1':
        updated_state = update_state_bisenetv1(state)
    elif model_type == 'pidnet':
        updated_state = update_state_pidnet(state)
    elif model_type == 'enet':
        updated_state = update_state_enet(state)
    elif model_type == 'ppliteseg':
        updated_state = update_state_ppliteseg(state)
    else:
        raise ValueError(f"incorrect model_type {model_type} specified")
    return updated_state


def add_mean_var_to_state(model_type, state, weight_mean, bias_mean,
                          weight_var, bias_var):
    """Add the variance parameters for last layer to model state
    """
    if model_type == 'bisenetv2':
        updated_state = add_var_bisenetv2(state,
                                          weight_mean,
                                          bias_mean,
                                          weight_var,
                                          bias_var)
    elif model_type == 'bisenetv1':
        updated_state = add_var_bisenetv1(state,
                                          weight_mean,
                                          bias_mean,
                                          weight_var,
                                          bias_var)
    elif model_type == 'pidnet':
        updated_state = add_mean_var_pidnet(state,
                                            weight_mean,
                                            bias_mean,
                                            weight_var,
                                            bias_var)
    elif model_type == 'enet':
        updated_state = add_mean_var_enet(state,
                                          weight_mean,
                                          bias_mean,
                                          weight_var,
                                          bias_var)
    elif model_type == 'ppliteseg':
        updated_state = add_mean_var_ppliteseg(state,
                                          weight_mean,
                                          bias_mean,
                                          weight_var,
                                          bias_var)
    return updated_state

def add_mean_var_pidnet(state, weight_mean, bias_mean,
                        weight_var, bias_var):
    state['final_layer.conv2.weight'] = weight_mean
    state['final_layer.conv2.bias'] = bias_mean
    state['final_layer.conv_var.weight'] = weight_var
    state['final_layer.conv_var.bias'] = bias_var
    return state

def add_mean_var_enet(state, weight_mean, bias_mean,
                        weight_var, bias_var):
    state['transposed_conv.weight'] = weight_mean
    state['out_var.weight'] = weight_var
    return state

def add_mean_var_ppliteseg(state, weight_mean, bias_mean,
                           weight_var, bias_var):
    state['seg_head.conv_out.weight'] = weight_mean
    state['seg_head.conv_var.weight'] = weight_var
    return state




def add_var_bisenetv2(state, weight_mean, bias_mean,
                      weight_var, bias_var):
    state['head.conv_out.weight'] = weight_mean
    state['head.conv_out.bias'] = bias_mean
    state['head.conv_var.weight'] = weight_var
    state['head.conv_var.bias'] = bias_var
    return state

def add_var_bisenetv1(state, weight_mean, bias_mean,
                      weight_var, bias_var):
    state['conv_out.conv_out.weight'] = weight_mean
    state['conv_out.conv_out.bias'] = bias_mean
    state['conv_out.conv_var.weight'] = weight_var
    state['conv_out.conv_var.bias'] = bias_var
    return state



def update_state_bisenetv2(state):
    # need to get  rid of the model prefix in the keys
    keys = list(state.keys())
    for key in keys:
        # if the key is for the augmentation stuff within the network, remove these vars.
        if 'conv_out.0.1' in key:
            new_key = key.replace('conv_out.0.1', 'conv_pre.1')
            state[new_key] = state.pop(key)
        elif 'conv_out.1' in key:
            new_key = key.replace('conv_out.1', 'conv_out')
            state[new_key] = state.pop(key)
        # elif 'conv_out' in key:
        # else:
        #     # otherwise want to keep the parameter and change the naming
        #     # convention so it will work appropriately
        #     new_key = key.replace('model.', '')
        #     state[new_key] = state.pop(key)
    return state

def update_state_bisenetv1(state):
    # actually don't need to do anything here for bisenetv1 :)
    return state

def update_state_enet(state):
    # actually don't need to do anything here for bisenetv1 :)
    state_dict = state['state_dict']
    # if state_dict['transposed_conv.weight'].shape[1] == 20:
    #     state_dict['transposed_conv.weight'] =  state_dict['transposed_conv.weight'][:, 1::, :, :]
    # print(state_dict['transposed_conv.weight'].shape)
    return state_dict


def update_state_pidnet(state):
    # need to get  rid of the model prefix in the keys
    keys = list(state.keys())
    for key in keys:
        # if the key is for the augmentation stuff within the network, remove these vars.
        if ('seghead_' in key) or ('loss.criterion' in key):
            # remove this param from the dict entirely
            _ = state.pop(key)
        else:
            # otherwise want to keep the parameter and change the naming
            # convention so it will work appropriately
            new_key = key.replace('model.', '')
            state[new_key] = state.pop(key)
    return state



def update_state_ppliteseg(state):
    return state




def ppliteseg_supergradients_to_mine(state):
    """converts supergradients params to my version"""
    # need to get  rid of the model prefix in the keys
    state = state['net']
    keys = list(state.keys())
    for key in keys:
        # otherwise want to keep the parameter and change the naming
        # convention so it will work appropriately
        new_key = key.replace('module.', '')
        new_key = new_key.replace('context_', 'context_module.')
        # replace my key for the final conv layer
        new_key = new_key.replace('seg_head.0.seg_head.2.weight', 'seg_head.conv_out.weight')
        new_key = new_key.replace('seg_head.0.seg_head.0.', 'seg_head.seg_head.0.')

        state[new_key] = state.pop(key)

    # remove aux head params
    keys = list(state.keys())
    for key in keys:
        if 'aux_heads' in key:
            del state[key]
    return state
