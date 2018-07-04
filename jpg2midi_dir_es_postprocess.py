import os
import sys
import numpy as np
import midi
from PIL import Image
from es_base import ES
from midi_reward import midi_score
import argparse

parser = argparse.ArgumentParser(description='Image Generation with GAN')
parser.add_argument('input', metavar='input', type=str, help='')
parser.add_argument('output', metavar='output', type=str, help='')
parser.add_argument('--std', type=float, default=0.1, required=False, help='')
parser.add_argument('--generations', type=int, default=500, required=False, help='')
parser.add_argument('--populations', type=int, default=500, required=False, help='')
parser.add_argument('--offsprings', type=int, default=200, required=False, help='')
parser.add_argument('--pi', type=int, default=4, required=False, help='')
parser.add_argument('--bar_multiplier', type=int, default=4, required=False, help='')
parser.add_argument('--phrase_multiplier', type=int, default=4, required=False, help='')
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
    dna = np.append(piano_roll.flatten(), 127.5).astype(np.float32)[np.newaxis,...].repeat(args.populations, axis=0)
    pop = dict(DNA=dna, mut_strength=np.random.randn(*dna.shape) * args.std)
    
    def F(x):
        x = x.copy()
        piano_roll_flattens, thresholds = x[...,:-1], x[...,-1]
        piano_rolls = piano_roll_flattens.reshape(piano_roll_flattens.shape[0],*shape)
        scores = []
        for i, piano_roll in enumerate(piano_rolls):
            piano_roll[piano_roll<thresholds[i]] = 0
            piano_roll[piano_roll>0] = 1
            piano_roll = piano_roll.astype(np.uint8)
            scores.append(midi_score(piano_roll, args.pi, args.bar_multiplier, args.phrase_multiplier))
        return np.asarray(scores)
    
    es = ES(fitness=F, dna_length=dna.shape[-1], bound=[0, 255], generations=args.generations,
        population_size=args.populations, offspring_size=args.offsprings, type='maximize')
    print('file: %d/%d'%(fn+1,len(dirs)))
    fitness = es.fit(pop, runs=1, show_progress=True)
    '''
    for f in range(args.generations):
        kids = es.get_offspring(pop)
        pop  = es.put_kids(pop, kids)
        pop  = es.selection(pop)
        fitness = F(pop['DNA'][-1:]).mean()
        print('[%d/%d] | [%d/%d] | fitness: %.2f'%(fn+1,len(dirs),f+1,args.generations,fitness))
    '''
    piano_roll, threshold = pop['DNA'][-1,:-1].reshape(shape), pop['DNA'][-1,-1]
    piano_roll = np.lib.pad(piano_roll, ((0,1),(0,1)), 'constant', constant_values=((0,0),(0,0)))
    piano_roll[piano_roll<threshold] = 0
    piano_roll[piano_roll>0] = 255
    piano_roll = piano_roll.astype(np.uint8)
    
    all_fitness.append(fitness)
    
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
tops = all_outputs[all_fitness.argsort()][:5]
with open('top5.txt', 'w') as fp:
    for top in tops:
        fp.write('%s\n'%str(top))
