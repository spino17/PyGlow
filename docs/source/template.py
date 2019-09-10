class Template:
    """
    Template demo.

    Arguments:
        a (int): param1
        b (int): param2
        c (int): param3
    """
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def func(self, segment):
        """
        Func method.

        Arguments:
            segment (iterable): list

        Returns:
            (int): a list of objects
        """
        return 2 * segment
