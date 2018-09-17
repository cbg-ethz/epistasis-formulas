import numpy as np
import pandas as pd


def compute_significant_masks(df):
    sign4_mask = df.fec_d_sign4.notna() | df.dev_sign4.notna() | df.CFU_sign4.notna() | df.time_d_sign4.notna()
    sign5_mask = df.fec_d_sign5_adj.notna() | df.dev_sign5.notna() | df.CFU_sign5.notna() | df.time_d_sign5.notna()
    return sign4_mask, sign5_mask

 
def make_tags(df):
    combinations = df['combination']
    setups = df['Unnamed: 1']
    c_tags = make_circuit_tags(combinations[:640], setups[:640])
    u_tags = make_u_tags(combinations[640:], setups[640:])
    return c_tags + u_tags


def make_circuit_tags(ids, setups):
    circuit_tags = [
        '{0:s}_{1:s}'.format(i.lower(), setup) for i, setup in zip(ids, setups)
    ]
    return circuit_tags


def make_u_tags(ids, setups):
    u3_tags = make_u3_tags(ids[:160], setups[:160])
    u4_tags = make_u4_tags(ids[160:270], setups[160:270])
    u5_tags = make_u5_tags(ids[270:], setups[270:])
    return u3_tags + u4_tags + u5_tags


def make_u3_tags(backgrounds, setups):
    return ['{0:s}_{1:.0f}'.format(s, float(b)) for b, s in zip(backgrounds, setups)]


def make_u4_tags(backgrounds, setups):
    return [
        make_u4_tag(background, setup)
        for background, setup in zip(backgrounds, setups)
    ]


def make_u4_tag(background, setup):
    u_tag = []
    for c, t in zip('abcd', setup[2:]):
        if t == '1':
            u_tag.append(c.upper())
        else:
            u_tag.append(c)

    i = 0
    while background[i] == '*':
        i += 1
    u_tag.insert(i, background[i])
    return 'u_' + ''.join(u_tag)


def make_u5_tags(backgrounds, setups):
    return [make_u5_tag(tag) for tag in setups]


def make_u5_tag(tag):
    u_tag = []
    for c, t in zip('abcde', tag[2:]):
        if t == '1':
            u_tag.append(c.upper())
        else:
            u_tag.append(c)
    return 'u_' + ''.join(u_tag)


if __name__ == '__main__':
    df = pd.read_csv('data_lisa.csv')

    # Tag data with pystasis tags
    tags = make_tags(df)
    df['tags'] = tags
    #print(df)

    # Select significant data
    s4, s5 = compute_significant_masks(df)
    df4 = df[s4]
    df5 = df[s5]
    print("df4",df4)
    #print(len(df4))
    print(len(df5))
