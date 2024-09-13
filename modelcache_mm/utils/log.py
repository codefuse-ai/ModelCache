# -*- coding: utf-8 -*-
import logging

FORMAT = '%(asctime)s - %(thread)d - %(filename)s-%(module)s:%(lineno)s - %(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT)

modelcache_log = logging.getLogger('modelcache')
