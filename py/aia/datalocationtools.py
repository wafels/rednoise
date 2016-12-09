import os


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
        if not(os.path.exists(loc)):
            os.makedirs(loc)
        locations[k] = loc
    return locations


def ident_creator(branches):
    return '_'.join(branches)


def filename(branches, separator='_'):
    """
    Creates a filename from a list of input strings.

    :param branches: a list of strings
    :param separator: string that separates the components strings
    :return: a single string with the list entries separated by the separator
    """
    return separator.join(branches)


def path(branches, root=''):
    """
    Creates a directory path from a list of input strings.

    :param branches: a list of strings.
    :param root: the root of directory path.
    :return: a single string with the list entries by the directory separator.
    """
    return os.path.join(root, os.path.sep.join(branches))


def filepath(branches, root='', separator='_'):
    """
    Creates a full filepath based on the input branches.

    :param branches:  a list of strings.
    :param root: the root of directory path.
    :return: a single string that defines a file path for a single file.
    """
    return os.path.join(path(branches, root=root),
                        filename(branches, separator=separator))


class FilePath:
    def __init__(self, branches, separator='_', root=''):
        self.branches = branches
        self.separator = separator
        self.root = root

        self.filename = filename(self.branches, separator=self.separator)
        self.path = path(self.branches, root=self.root)
        self.filepath = filepath(self.branches, root=self.root, separator=self.separator)
