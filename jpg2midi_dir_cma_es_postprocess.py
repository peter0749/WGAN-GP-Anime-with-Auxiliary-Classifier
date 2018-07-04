import os
import sys
import numpy as np
import midi
from PIL import Image
from cma import CMAEvolutionStrategy as ES
from midi_reward import midi_score
import argparse

parser = argparse.ArgumentParser(description='Image Generation with GAN')
parser.add_argument('input', metavar='input', type=str, help='')
parser.add_argument('output', metavar='output', type=str, help='')
parser.add_argument('--generations', type=int, default=500, required=False, help='')
parser.add_argument('--sigma', type=float, default=5, required=False, help='')
parser.add_argument('--pi', type=int, default=4, required=False, help='')
parser.add_argument('--bar_multiplier', type=int, default=4, required=False, help='')
parser.add_argument('--phrase_multiplier', type=int, default=4, required=False, help='')
parser.add_argument('--dice_original', type=float, default=0.2, required=False, help='')
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

all_fitness = []
all_outputs = []
dirs = os.listdir(args.input)
for fn, filename in enumerate(dirs):
    fullpath = args.input+'/'+filename
    piano_roll = np.array(Image.open(fullpath))
    piano_roll = np.transpose(np.flip(piano_roll, 0), (1,0))
    
    shape = piano_roll.shape
    dna = np.append(piano_roll.flatten(), 127.5).astype(np.float32)
    
    def F(orignal_piano_roll, dice_w=1.0):
        orignal_piano_roll_t = orignal_piano_roll.copy()
        orignal_piano_roll_t[orignal_piano_roll_t<127.5] = 0
        orignal_piano_roll_t[orignal_piano_roll_t>0    ] = 1
        orignal_piano_roll_t = orignal_piano_roll_t.astype(np.uint8)
        def F_block(x):
            x = x.copy()
            piano_roll_flattens, thresholds = x[...,:-1], x[...,-1]
            piano_rolls = piano_roll_flattens.reshape(piano_roll_flattens.shape[0],*shape)
            scores = []
            for i, piano_roll in enumerate(piano_rolls):
                piano_roll[piano_roll<thresholds[i]] = 0
                piano_roll[piano_roll>0] = 1
                piano_roll = piano_roll.astype(np.uint8)
                score  = midi_score(piano_roll, args.pi, args.bar_multiplier, args.phrase_multiplier)
                y_t, y_p = orignal_piano_roll_t.flatten(), piano_roll.flatten()
                score += dice_w * np.mean( 2*y_t*y_p / (y_t.sum()+y_p.sum()+1) ) # bce (prevent edit too far away from orignal piece)
                scores.append(score)
            return np.asarray(scores)
        return F_block
    fitness_func = F(piano_roll, args.dice_original)
    es = ES(dna,args.sigma)
    for f in range(args.generations):
        dnas = np.asarray(es.ask())
        es.tell(dnas, fitness_func(dnas))
        es.disp()
    es.result_pretty()
    
    dna = np.array(es.result[0])
    #print(dna.shape)
    piano_roll, threshold = dna[:-1].reshape(shape), dna[-1]
    piano_roll = np.lib.pad(piano_roll, ((0,1),(0,1)), 'constant', constant_values=((0,0),(0,0)))
    piano_roll[piano_roll<threshold] = 0
    piano_roll[piano_roll>0] = 255
    piano_roll = piano_roll.astype(np.uint8)
    
    score = midi_score(piano_roll, args.pi, args.bar_multiplier, args.phrase_multiplier)
    if np.sum(piano_roll>0) / float(np.sum(piano_roll>=0)) < 0.005:
        score -= 1e6
    all_fitness.append(score)
    
    pattern = midi.Pattern(resolution=4)
    track = midi.Track()
    pattern.append(track)
    lastEvent = 0
    sweep = np.zeros_like(piano_roll[0], dtype=np.bool)
    for tick, line in enumerate(piano_roll):
        for i, note in enumerate(line):
            if note>0 and not sweep[i]: # a note on event
                track.append(midi.NoteOnEvent(tick=tick-lastEvent, data=[i+34 ,127]))
                lastEvent = tick
                sweep[i]=True
            if note==0 and sweep[i]: # a note off event
                track.append(midi.NoteOffEvent(tick=tick-lastEvent, data=[i+34 ,0]))
                lastEvent = tick
                sweep[i]=False
    track.append(midi.EndOfTrackEvent(tick=1))
    outpath = args.output+'/'+filename+'.mid'
    all_outputs.append(outpath)
    midi.write_midifile(outpath, pattern)
all_fitness = -np.asarray(all_fitness)
all_outputs = np.array(all_outputs)
good_idx = all_fitness.argsort()
tops = all_outputs[good_idx][:5]
top_scores = -all_fitness[good_idx][:5]
with open('top5.txt', 'w') as fp:
    for filename, score in zip(tops, top_scores):
        fp.write('%s, %.2f\n'%(str(filename), score))
