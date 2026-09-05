#!/usr/bin/env python3
# coding: utf-8

import time 
import datetime



def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        func_name = func.__name__
        print(f"Function: {func_name}; started at: {start_time_str}")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60.
        print(f"Function: {func_name}; finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed time: {elapsed_time:.2f} minutes")
        
        return result
    return wrapper
