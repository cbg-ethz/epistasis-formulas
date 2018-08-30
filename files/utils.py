"""Various useful utilities"""

import numpy as np


DATA_TO_PYSTASIS_PERM = (
    0b00000,
    0b10000,
    0b01000,
    0b00100,
    0b00010,
    0b00001,
    0b11000,
    0b10100,
    0b10010,
    0b10001,
    0b01100,
    0b01010,
    0b01001,
    0b00110,
    0b00101,
    0b00011,
    0b11100,
    0b11010,
    0b11001,
    0b10110,
    0b10101,
    0b10011,
    0b01110,
    0b01101,
    0b01011,
    0b00111,
    0b11110,
    0b11101,
    0b11011,
    0b10111,
    0b01111,
    0b11111
)


def convert_vector_to_pystasis_order(w):
    out_w = np.empty_like(w)
    for i, e in enumerate(w):
        out_w[DATA_TO_PYSTASIS_PERM[i]] = e
    return out_w


def convert_dataframe_to_pystasis_order(df):
    return df.set_index(
        np.array(DATA_TO_PYSTASIS_PERM),
        verify_integrity=True
    ).sort_index()


def is_standard_tag(tag):
    if tag[0] in 'aceu':
        setup = tag[2:]
        if not '1' in setup and all(c.isupper() for c in setup if c.isalpha()):
            return True
    return False


def split_pos_neg(a):
    abs_a = np.absolute(a)
    pos = (abs_a + a) // 2
    neg = a - pos
    return pos, neg


def format_context(context, n):
    fmtstr = ['*'] * n
    for i, v in context:
        fmtstr[i] = str(v)

    j = 0
    for i in range(n):
        if fmtstr[i] == '*':
            fmtstr[i] = chr(ord('A') + j)
            j += 1

    return ''.join(fmtstr)

def modulate_tag(tag, n, i):
    br = np.binary_repr(i, width=n)
    b_idx = 0
    l_tag = list(tag)
    for i, c in enumerate(l_tag):
        if c.isalpha():
            if br[b_idx] == '0':
                l_tag[i] = c.lower()
            else:
                l_tag[i] = c.upper()
            b_idx += 1
    return ''.join(l_tag)
