import pstats
from pstats import SortKey

p1 = pstats.Stats('cProfile_init')
p1.strip_dirs().sort_stats('cumulative').print_stats()

p2 = pstats.Stats('cProfile_step')
p2.strip_dirs().sort_stats('cumulative').print_stats()
