import sys 

def is_debugger_active() -> bool:
    gettrace = getattr(sys, 'gettrace', lambda : None)
    return gettrace() is not None