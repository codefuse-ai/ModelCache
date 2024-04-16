# -*- coding: utf-8 -*-


def get_index_name(model):
    return 'modelcache' + '_' + model


def get_index_prefix(model):
    return 'prefix' + '_' + model


def get_mm_index_name(model, iat_type):
    if iat_type not in ['IMG_TEXT', 'iat', 'IMG', 'image', 'TEXT', 'text']:
        raise ValueError('iat_type is not normal!')
    if iat_type == 'IMG_TEXT':
        iat_type = 'iat'
    elif iat_type == 'IMG':
        iat_type = 'image'
    elif iat_type == 'TEXT':
        iat_type = 'text'
    return 'multicache' + '_' + model + '_' + iat_type


def get_collection_iat_prefix(model, iat_type, table_suffix):
    if iat_type not in ['IMG_TEXT', 'iat', 'IMG', 'image', 'TEXT', 'text']:
        raise ValueError('iat_type is not normal!')
    if iat_type == 'IMG_TEXT':
        iat_type = 'iat'
    elif iat_type == 'IMG':
        iat_type = 'image'
    elif iat_type == 'TEXT':
        iat_type = 'text'
    return 'prefix' + '_' + model + '_' + iat_type + '_' + table_suffix