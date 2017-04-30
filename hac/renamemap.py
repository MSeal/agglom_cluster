from collections import namedtuple

# Separate this definition from the cython file so pickling can find the class
RenameMapping = namedtuple('RenameMapping', ['integer', 'original', 'max_node'])
