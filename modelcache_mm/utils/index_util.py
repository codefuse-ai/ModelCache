# -*- coding: utf-8 -*-


def get_index_name(model):
    return 'modelcache' + '_' + model


def get_index_prefix(model):
    return 'prefix' + '_' + model


def get_mm_index_name(model, mm_type):
    print('mm_type: {}'.format(mm_type))
    if mm_type not in ['IMG_TEXT', 'mm', 'IMG', 'image', 'TEXT', 'text']:
        raise ValueError('mm_type is not normal!')
    if mm_type == 'IMG_TEXT':
        mm_type = 'mm'
    elif mm_type == 'IMG':
        mm_type = 'image'
    elif mm_type == 'TEXT':
        mm_type = 'text'
    return 'multicache' + '_' + model + '_' + mm_type


def get_mm_index_prefix(model, mm_type):
    if mm_type not in ['IMG_TEXT', 'mm', 'IMG', 'image', 'TEXT', 'text']:
        print('mm_type: {}'.format(mm_type))
        raise ValueError('mm_type is not normal!')
    if mm_type == 'IMG_TEXT':
        mm_type = 'mm'
    elif mm_type == 'IMG':
        mm_type = 'image'
    elif mm_type == 'TEXT':
        mm_type = 'text'
    return 'prefix' + '_' + model + '_' + mm_type
