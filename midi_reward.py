"""
0,  unison_perfect,  +15
1,  second_minor,    -100
2,  second_major,    -50
3,  third_minor,     +5
4,  third_major,     +4
5,  forth_perfect,   +10
6,  forth_augmented, -8
7,  fifth_perfect,   +10
8,  fifth_minor,      0
9,  sixth_major,     +4
10, sixth_minor,     +3
11, seventh_major,   -10
12, octave_perfect,  +12
"""
score_table = [15,-100,-50,5,4,10,-8,10,0,4,3,-10,12]
perfects = set([0, 5, 7, 12])
near_perfect = set([3,4,9,10])
consonant = perfects or near_perfect
dissonant = set([1,2,6,11])
import numpy as np

def midi_score(piano_roll, pi_=4, bar_multiplier=4, phrase_multiplier=4): # shape: (ts, pitch)
    intra_note_score = 0.0
    inter_note_score = 0.0
    playable = 0.0
    tempo_score = 0.0
    stable_score = 0.0
    note_density_score = 0.0
    prev_root = None
    prev_head = None
    prev_interval = None
    last_notes = None
    bar = pi_*bar_multiplier
    phrase = bar*phrase_multiplier
    for tick in range(0, len(piano_roll), phrase):
        segment = piano_roll[tick:tick+phrase]
        natural = np.where(segment>0)[1] % 12
        hist_std = np.histogram(natural, bins=np.arange(12), range=(0,12), normed=True)[0].std() * 3
        stable_score -= hist_std
        
    for tick, notes in enumerate(piano_roll):
        new_notes = np.array(notes>0) if last_notes is None else np.array((notes>0) & (last_notes<=0))
        last_notes = notes
        note_index = np.where(new_notes>0)[0]
        #note_index = np.where(notes>0)[0]
        if len(note_index)==0:
            intra_note_score += 0 if (not prev_interval is None) and (prev_interval in consonant) else -3
            continue
        if tick%pi_!=0:
            tempo_score -= 3
        
        if len(note_index)>6:
            playable -= 50 
        root = note_index[0]
        head = note_index[-1]
        
        # compute intra interval scores ( O(n^2) ?! this should be an issue... )
        for i in range(len(note_index)-1):
            for j in range(i+1, len(note_index)):
                a = note_index[i]
                b = note_index[j]
                diff = b-a
                if diff>24:
                    intra_note_score -= 30
                elif diff<12:
                    intra_note_score += score_table[int(diff)]
        
        curr_interval = note_index[1]-note_index[0] if len(note_index)>=2 else None
        '''
        # compute inter interval scores:
        if not prev_root is None:
            if len(note_index)>1:
                inter_note_score += score_table[int(np.abs(prev_root-root))] if int(np.abs(prev_root-root))<=12 else -100
            inter_note_score += score_table[int(np.abs(prev_head-head))] if int(np.abs(prev_head-head))<=12 else -200
            if (not curr_interval is None) and (not prev_interval is None):
                if curr_interval in dissonant:
                    if prev_interval in dissonant: # bad bad bad
                        inter_note_score -= 5
                    elif prev_interval in perfects: # transition
                        inter_note_score += 5
                    elif prev_interval in near_perfect: # not bad
                        inter_note_score += 3
                elif curr_interval in perfects:
                    if prev_interval in dissonant: # perfect
                        inter_note_score += 8
                    elif prev_interval in perfects: # boring
                        inter_note_score -= 3
                    elif prev_interval in near_perfect: # a little boring
                        inter_note_score += 3
                elif curr_interval in near_perfect:
                    if prev_interval in dissonant: # good
                        inter_note_score += 5
                    elif prev_interval in perfects:
                        inter_note_score += 1
                    elif prev_interval in near_perfect:
                        inter_note_score -= 3
        '''
        prev_root = root
        prev_head = head
        prev_interval = curr_interval
        a = (piano_roll>0).sum()
        b = (piano_roll==0).sum()
        if a / (a+b) < 0.03:
            note_density_score -= 100
    return np.nan_to_num(intra_note_score + inter_note_score + playable + tempo_score + stable_score + note_density_score)