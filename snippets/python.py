# Dictionary
from pprint import pprint
import toolz

# Create a new dictionary, sorted by the keys alphabetically
# same key:val pairs, only the order in the dictionary is changed so that key:val with 
# alphabetically-first key comes first
my_dict = {
    'b':10,
    'k':100,
    'c':20,
    'a':70
}
print('original: ', my_dict)

sorted_dict = dict(sorted(my_dict.items()))
print('sorted: ', sorted_dict)

# Use toolz lib. for conveninent and optimized 'map'ping on dictionaries
def add_10(x): return x+10
print('original: ', my_dict)
print('after applying add_10 func: ')
pprint(toolz.valmap(add_10, my_dict))

