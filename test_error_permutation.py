#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  测试 error 和 Permutation的关系
  Created:  04/19/17
"""
import os
import tensorflow as tf

permutation = range(0,22,2)

for p_num in permutation:
    os.system("python3 Hierarchy_GAN_TOPOLOGY_MAIN.py --training_step 1000 --permutation_step %d --Dataset facebook"%p_num)