#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 11:05:07 2021

@author: renuka
"""
import cProfile
import pstats
from main import main

def stats():
    cProfile.run("main()", "restats")
    p = pstats.Stats('restats') 
    p.sort_stats('cumulative').print_stats(20)
    # Change the number if you want to visualize more stats

stats()
    
