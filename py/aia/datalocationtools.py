import os

###############################################################################
# Save location calculators
#
def location_branch(location_root, branches):
    """Recursively adds a branch to a directory listing"""
    loc = os.path.expanduser(location_root)
    for branch in branches:
        loc = os.path.join(loc, branch)
    return loc


def save_location_calculator(roots, branches):
    """Takes a bunch of roots and creates subdirectories as needed"""
    locations = {}
    for k in roots.keys():
        loc = location_branch(roots[k], branches)
        os.makedirs(loc)
        locations[k] = loc
    return locations


def ident_creator(branches):
    return '_'.join(branches)

