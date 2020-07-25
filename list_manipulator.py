'''
pop_all

# Pop entire list
@params {list} list to be popped
'''
# Pop all in array
def pop_all(l):
    r, l[:] = l[:], []
    return r